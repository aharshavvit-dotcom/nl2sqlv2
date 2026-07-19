"""Offline Failure Attribution & Route Diagnostics Runner.

Implements Stage 2 of Semantic-First Production Hardening.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from evaluation.report_schemas import (
    RouteDiagnosticReport,
    RouteDiagnosticCase,
    RoutePredictionResult,
    RendererAttributionResult,
)
from evaluation.semantic_pass import compute_simple_query_semantic_pass
from ir.query_ir_models import diff_query_ir
from inference import (
    RuntimeMode,
    PredictionRoute,
    DiagnosticContext,
    DiagnosticRoutingNotAllowedError,
)
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from nl2sql_v1.schema import SchemaGraph
from training.evaluate_generic_models import _schema_graph


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic execution for route diagnostics (Comment #19)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_dataset_hash(records: list[dict[str, Any]]) -> str:
    """Compute deterministic hash of the evaluation dataset."""
    hasher = hashlib.sha256()
    for r in sorted(records, key=lambda x: str(x.get("example_id", ""))):
        hasher.update(str(r.get("example_id", "")).encode("utf-8"))
        hasher.update(str(r.get("question", "")).encode("utf-8"))
    return hasher.hexdigest()[:16]


def attribute_failure_stage(
    route: str,
    gold_ir: dict[str, Any],
    pred_result: Any,
    available: bool,
    reason: str | None
) -> str | None:
    """Determine the hierarchical stage of failure (Comment #7)."""
    if not available:
        return "route_unavailable"

    debug = pred_result.debug or {}
    boundaries = debug.get("boundaries") or {}
    native_ir = boundaries.get("native_query_ir") or pred_result.query_ir or {}
    resolved_ir = boundaries.get("resolved_query_ir") or pred_result.query_ir or {}
    validated_ir = boundaries.get("validated_query_ir") or pred_result.query_ir or {}
    rendered_sql = boundaries.get("rendered_sql") or pred_result.sql
    sql_validation = boundaries.get("sql_validation") or pred_result.validation or {}

    if not native_ir:
        return "route_generation_failure"

    # 1. Native QueryIR semantic match
    native_diff = diff_query_ir(native_ir, gold_ir)
    native_correct = all(v is True for k, v in native_diff.items() if k.endswith("_match"))
    if not native_correct:
        return "query_ir_semantic_failure"

    # 2. Resolved QueryIR semantic match (check slot resolution)
    resolved_diff = diff_query_ir(resolved_ir, gold_ir)
    resolved_correct = all(v is True for k, v in resolved_diff.items() if k.endswith("_match"))
    if not resolved_correct:
        return "slot_resolution_failure"

    # 3. Validated QueryIR validation checks
    validated_diff = diff_query_ir(validated_ir, gold_ir)
    validated_correct = all(v is True for k, v in validated_diff.items() if k.endswith("_match"))
    if not validated_correct:
        return "query_ir_validation_failure"

    # 4. Renderer SQL generation checks
    if not rendered_sql:
        return "renderer_generation_failure"

    # 5. SQL Validation check
    if not sql_validation.get("is_valid", sql_validation.get("ok", False)):
        # Check if sql validator repair was attempted
        repair = debug.get("sql_repair") or {}
        if repair.get("repair_attempted") and not repair.get("repair_succeeded"):
            return "sql_repair_failure"
        return "sql_validation_failure"

    # 6. Database execution checks (execution result checking)
    exec_res = boundaries.get("execution_result") or {}
    if exec_res.get("failed"):
        return "database_execution_failure"

    # 7. Semantic execution result comparison
    if exec_res.get("value_mismatch"):
        return "result_semantic_mismatch"

    return None


def run_renderer_control(
    example_id: str,
    question: str,
    gold_ir: dict[str, Any],
    pred_result: Any,
    model: RetrievalNL2SQLModel,
    schema: Any,
) -> RendererAttributionResult:
    """Control experiment for renderer bugs vs model representation (Comment #8)."""
    dialect = model.orchestrator.sql_renderer.dialect if hasattr(model.orchestrator.sql_renderer, "dialect") else "sqlite"
    
    # 1. Try rendering predicted IR
    pred_ir = pred_result.query_ir or {}
    pred_render_success = False
    pred_rendered_sql = None
    pred_val = {}
    if pred_ir:
        try:
            pred_rendered_sql = model.orchestrator.sql_renderer.render(pred_ir, dialect=dialect)
            repair_pred = model.orchestrator.sql_validator.validate_with_repair(
                pred_rendered_sql, schema=schema, max_limit=1000, dialect=dialect
            )
            pred_val = repair_pred.get("final_validation") or {}
            pred_render_success = bool(pred_val.get("is_valid", pred_val.get("ok", False)))
        except Exception as e:
            pred_val = {"is_valid": False, "error": str(e)}

    # 2. Try rendering gold IR
    gold_render_success = False
    gold_rendered_sql = None
    gold_val = {}
    try:
        gold_rendered_sql = model.orchestrator.sql_renderer.render(gold_ir, dialect=dialect)
        repair_gold = model.orchestrator.sql_validator.validate_with_repair(
            gold_rendered_sql, schema=schema, max_limit=1000, dialect=dialect
        )
        gold_val = repair_gold.get("final_validation") or {}
        gold_render_success = bool(gold_val.get("is_valid", gold_val.get("ok", False)))
    except Exception as e:
        gold_val = {"is_valid": False, "error": str(e)}

    # 3. Assess matching & meaning
    pred_ir_diff = diff_query_ir(pred_ir, gold_ir)
    pred_ir_correct = all(v is True for k, v in pred_ir_diff.items() if k.endswith("_match"))

    if pred_ir_correct:
        if not pred_render_success:
            if not gold_render_success:
                meaning = "dialect_integration_failure"
                failure_stage = "schema_or_dialect_failure"
            else:
                meaning = "renderer_edge_case"
                failure_stage = "renderer_generation_failure"
        else:
            meaning = "success"
            failure_stage = "none"
    else:
        meaning = "predicted_ir_mismatch_not_renderer_failure"
        failure_stage = "query_ir_semantic_failure"

    return RendererAttributionResult(
        example_id=example_id,
        question=question,
        predicted_ir_correct=pred_ir_correct,
        gold_ir_render_success=gold_render_success,
        gold_ir_rendered_sql=gold_rendered_sql,
        gold_ir_sql_validation=gold_val,
        predicted_ir_render_success=pred_render_success,
        predicted_ir_rendered_sql=pred_rendered_sql,
        predicted_ir_sql_validation=pred_val,
        meaning=meaning,
        failure_stage=failure_stage,
    )


def compute_slot_matches(pred_ir: dict[str, Any], gold_ir: dict[str, Any]) -> dict[str, bool]:
    """Compare specific slot fields for route comparisons (Comment #13)."""
    diff = diff_query_ir(pred_ir, gold_ir)
    return {
        "intent_match": bool(diff.get("intent_match", True) is True),
        "base_table_match": bool(diff.get("base_table_match", True) is True),
        "projection_exact_match": bool(diff.get("projection_match", True) is True),
        "dimension_match": bool(diff.get("dimension_match", True) is True),
        "filter_column_match": bool(diff.get("filter_column_match", True) is True),
        "filter_value_match": bool(diff.get("filter_value_match", True) is True),
        "aggregation_match": bool(diff.get("aggregation_match", True) is True),
        "join_match": bool(diff.get("join_match", True) is True),
    }


def segment_metrics(cases: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    """Group statistics by a segment (Comment #14)."""
    segmented: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        val = str(c.get(group_key) or "unknown")
        segmented.setdefault(val, []).append(c)

    report = {}
    for segment, items in segmented.items():
        total = len(items)
        regrets = sum(1 for item in items if item.get("router_regret"))
        selected_passes = sum(1 for item in items if item.get("selected_route_passed"))
        oracle_passes = sum(1 for item in items if item.get("oracle_route_available"))
        
        # Retrieval-only and Neural-only passes
        retrieval_only = 0
        neural_only = 0
        both = 0
        neither = 0
        for item in items:
            routes = item.get("route_results") or {}
            ret_pass = routes.get("retrieval", {}).get("semantic_pass", False)
            neu_pass = routes.get("neural", {}).get("semantic_pass", False)
            if ret_pass and not neu_pass:
                retrieval_only += 1
            elif neu_pass and not ret_pass:
                neural_only += 1
            elif ret_pass and neu_pass:
                both += 1
            else:
                neither += 1

        report[segment] = {
            "total_cases": total,
            "selected_semantic_pass_rate": round(selected_passes / total, 4) if total else 0.0,
            "oracle_semantic_pass_rate": round(oracle_passes / total, 4) if total else 0.0,
            "router_regret_count": regrets,
            "router_regret_rate": round(regrets / total, 4) if total else 0.0,
            "neither_route_correct_rate": round(neither / total, 4) if total else 0.0,
            "route_distribution": {
                "retrieval_only": retrieval_only,
                "neural_only": neural_only,
                "both": both,
                "neither": neither,
            }
        }
    return report


def run_diagnostics(
    dataset_path: Path,
    artifact_dir: Path,
    neural_model_dir: Path | None,
    output_dir: Path,
    pipeline_run_id: str,
    limit: int | None = None,
) -> None:
    """Run diagnostics offline across all routes."""
    set_seed()

    # Load Model
    print(f"Loading RetrievalNL2SQLModel from {artifact_dir}...")
    model = RetrievalNL2SQLModel.load(
        artifact_dir=artifact_dir,
        neural_ir_model_dir=neural_model_dir,
        allow_dev_fallback=False,
    )

    # Load frozen dataset
    print(f"Reading dataset from {dataset_path}...")
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit:
        records = records[:limit]
        print(f"Limiting to first {limit} records.")

    dataset_hash = compute_dataset_hash(records)

    cases_output_path = output_dir / "route_diagnostics_cases.jsonl"
    renderer_output_path = output_dir / "renderer_attribution_cases.jsonl"

    print("Beginning offline forced routing predictions...")
    diagnostic_cases = []
    renderer_cases = []

    total_records = len(records)
    for idx, record in enumerate(records):
        example_id = str(record.get("example_id") or f"ex_{idx}")
        question = record.get("question", "")
        gold_ir = record.get("query_ir") or {}
        gold_sql = record.get("rendered_sql") or record.get("source_sql") or ""
        dataset = record.get("dataset_name") or record.get("dataset") or "standard"
        complexity = record.get("complexity") or "medium"
        intent = gold_ir.get("intent")

        schema_raw = record.get("schema") or {}
        schema_graph = _schema_graph(schema_raw)

        # 1. Structured Leakage check (Comment #10): Pred functions get only question & schema
        routes_to_evaluate = [
            ("automatic", None),
            ("direct_planner", PredictionRoute.DIRECT_PLANNER),
            ("retrieval", PredictionRoute.RETRIEVAL),
            ("neural", PredictionRoute.NEURAL),
        ]

        route_results = {}
        renderer_cases = []
        automatic_pred_result = None

        for route_name, forced_route in routes_to_evaluate:
            ctx = DiagnosticContext(
                forced_route=forced_route,
                cache_read_enabled=False,
                cache_write_enabled=False,
                telemetry_namespace="disabled",
                feedback_enabled=False,
                persist_runtime_state=False,
                runtime_mode=RuntimeMode.TEST,
            )

            available = True
            reason = None
            pred_result = None

            try:
                pred_result = model.predict(
                    question=question,
                    schema=schema_graph,
                    diagnostic_context=ctx,
                )
                if route_name == "automatic":
                    automatic_pred_result = pred_result
                if pred_result.debug.get("forced_route_unavailable"):
                    available = False
                    reason = pred_result.debug.get("unavailable_reason")
            except Exception as e:
                available = False
                reason = f"exception: {e}"

            if not available or pred_result is None:
                route_results[route_name] = RoutePredictionResult(
                    route=route_name,
                    available=False,
                    unavailable_reason=reason,
                )
                continue

            # Compute semantic pass
            validation = pred_result.validation or {}
            sem_pass = compute_simple_query_semantic_pass(
                gold_ir=gold_ir,
                predicted_ir=pred_result.query_ir,
                final_sql=pred_result.sql,
                validation_result=validation,
            )

            failure_stage = attribute_failure_stage(route_name, gold_ir, pred_result, available, reason)

            debug = pred_result.debug or {}
            boundaries = debug.get("boundaries") or {}

            route_results[route_name] = RoutePredictionResult(
                route=route_name,
                available=True,
                native_query_ir=boundaries.get("native_query_ir"),
                resolved_query_ir=boundaries.get("resolved_query_ir"),
                validated_query_ir=boundaries.get("validated_query_ir"),
                rendered_sql=boundaries.get("rendered_sql"),
                sql_validation=boundaries.get("sql_validation"),
                semantic_pass=bool(sem_pass.passed),
                failure_stage=failure_stage,
            )

            # Renderer Control experiment only for selected automatic/final prediction
            if route_name == "automatic":
                ctrl = run_renderer_control(example_id, question, gold_ir, pred_result, model, schema_graph)
                renderer_cases.append(ctrl)

        # Oracle calculations
        passing = [r for r, res in route_results.items() if res.semantic_pass and r != "automatic"]
        
        selected_route = "automatic"
        if automatic_pred_result is not None:
            router_dec = automatic_pred_result.debug.get("router_decision") or {}
            selected_route = router_dec.get("selected") or "automatic"
            
        if selected_route == "retrieval_ir":
            selected_route = "retrieval"
        if selected_route == "neural_ir":
            selected_route = "neural"

        selected_route_passed = route_results.get(selected_route, route_results["automatic"]).semantic_pass
        oracle_avail = len(passing) > 0
        regret = oracle_avail and not selected_route_passed

        case_diag = RouteDiagnosticCase(
            example_id=example_id,
            question=question,
            dataset=dataset,
            complexity=complexity,
            intent=intent,
            route_results=route_results,
            selected_route=selected_route,
            passing_routes=passing,
            selected_route_passed=selected_route_passed,
            oracle_route_available=oracle_avail,
            router_regret=regret,
        )
        diagnostic_cases.append(case_diag)

        # Write progress every 25 cases (Comment #20)
        if (idx + 1) % 25 == 0 or (idx + 1) == total_records:
            print(f"Processed {idx + 1}/{total_records} cases...")

    # Write case lists (Comment #15 JSONL format)
    print(f"Writing cases to {cases_output_path}...")
    with cases_output_path.open("w", encoding="utf-8") as f:
        for c in diagnostic_cases:
            f.write(c.model_dump_json() + "\n")

    print(f"Writing renderer cases to {renderer_output_path}...")
    with renderer_output_path.open("w", encoding="utf-8") as f:
        for r in renderer_cases:
            f.write(r.model_dump_json() + "\n")

    # Generate segmentation maps
    by_intent = segment_metrics([c.model_dump() for c in diagnostic_cases], "intent")
    by_dataset = segment_metrics([c.model_dump() for c in diagnostic_cases], "dataset")
    by_complexity = segment_metrics([c.model_dump() for c in diagnostic_cases], "complexity")

    # Write segmentation reports
    with (output_dir / "route_metrics_by_intent.json").open("w", encoding="utf-8") as f:
        json.dump(by_intent, f, indent=2)
    with (output_dir / "route_metrics_by_dataset.json").open("w", encoding="utf-8") as f:
        json.dump(by_dataset, f, indent=2)
    with (output_dir / "route_metrics_by_complexity.json").open("w", encoding="utf-8") as f:
        json.dump(by_complexity, f, indent=2)

    # Compute overall metrics
    total = len(diagnostic_cases)
    oracle_passes = sum(1 for c in diagnostic_cases if c.oracle_route_available)
    selected_passes = sum(1 for c in diagnostic_cases if c.selected_route_passed)
    regrets = sum(1 for c in diagnostic_cases if c.router_regret)

    # Neither-route-correct rate
    neither = sum(
        1 for c in diagnostic_cases
        if not (c.route_results.get("retrieval") and c.route_results["retrieval"].semantic_pass)
        and not (c.route_results.get("neural") and c.route_results["neural"].semantic_pass)
        and not (c.route_results.get("direct_planner") and c.route_results["direct_planner"].semantic_pass)
    )

    # Accumulate failure stages
    stages = {}
    for c in diagnostic_cases:
        r_res = c.route_results.get(c.selected_route) or c.route_results.get("automatic")
        stage = r_res.failure_stage if r_res else None
        if stage:
            stages[stage] = stages.get(stage, 0) + 1

    route_distribution = dict(Counter(c.selected_route for c in diagnostic_cases))
    route_percentage = {
        route: round(count / total, 4) if total else 0.0
        for route, count in sorted(route_distribution.items())
    }
    unique_wins: dict[str, int] = {}
    unique_regressions: dict[str, int] = {}
    route_names = ["direct_planner", "retrieval", "neural"]
    for route_name in route_names:
        wins = 0
        regressions = 0
        for c in diagnostic_cases:
            route_result = c.route_results.get(route_name)
            if route_result is None or not route_result.available:
                continue
            other_results = [
                result for name, result in c.route_results.items()
                if name in route_names and name != route_name and result.available
            ]
            if route_result.semantic_pass and not any(result.semantic_pass for result in other_results):
                wins += 1
            if c.selected_route == route_name and not route_result.semantic_pass and any(
                result.semantic_pass for result in other_results
            ):
                regressions += 1
        unique_wins[route_name] = wins
        unique_regressions[route_name] = regressions

    # Accumulate slot-level matches per route
    slots_acc = {}
    for rname in ["direct_planner", "retrieval", "neural"]:
        r_cases = [c for c in diagnostic_cases if c.route_results.get(rname) and c.route_results[rname].available]
        if not r_cases:
            continue
        slots_acc[rname] = {}
        for sfield in ["intent_match", "base_table_match", "projection_exact_match", "dimension_match", "filter_column_match", "filter_value_match", "aggregation_match", "join_match"]:
            count = 0
            for c in r_cases:
                pred_ir = c.route_results[rname].validated_query_ir or {}
                # retrieve gold_ir matching this example
                match_rec = next(x for x in records if str(x.get("example_id")) == c.example_id)
                gold_ir = match_rec.get("query_ir") or {}
                matches = compute_slot_matches(pred_ir, gold_ir)
                if matches.get(sfield):
                    count += 1
            slots_acc[rname][sfield] = round(count / len(r_cases), 4)

    summary = {
        "total_cases": total,
        "selected_route_semantic_pass_rate": round(selected_passes / total, 4) if total else 0.0,
        "best_of_routes_semantic_pass_rate": round(oracle_passes / total, 4) if total else 0.0,
        "router_oracle_gap": round((oracle_passes - selected_passes) / total, 4) if total else 0.0,
        "router_regret_count": regrets,
        "router_regret_rate": round(regrets / total, 4) if total else 0.0,
        "neither_route_correct_rate": round(neither / total, 4) if total else 0.0,
        "failure_stages_breakdown": stages,
        "route_distribution": route_distribution,
        "route_percentage": route_percentage,
        "unique_wins_by_route": unique_wins,
        "unique_regressions_by_selected_route": unique_regressions,
        "per_slot_accuracy_by_route": slots_acc,
    }

    # Build schema-validated RouteDiagnosticReport (Comment #16)
    report = RouteDiagnosticReport(
        pipeline_run_id=pipeline_run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset_hash=dataset_hash,
        summary=summary,
        by_dataset=by_dataset,
        by_intent=by_intent,
        by_complexity=by_complexity,
    )

    report_output_path = output_dir / "route_diagnostics_summary.json"
    print(f"Writing final diagnostics summary report to {report_output_path}...")
    report_output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print("=== Diagnostics Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--neural-model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pipeline-run-id", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds_path = Path(args.dataset_path)
    art_dir = Path(args.artifact_dir)
    neu_dir = Path(args.neural_model_dir) if args.neural_model_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_diagnostics(
        dataset_path=ds_path,
        artifact_dir=art_dir,
        neural_model_dir=neu_dir,
        output_dir=out_dir,
        pipeline_run_id=args.pipeline_run_id,
        limit=args.limit,
    )
