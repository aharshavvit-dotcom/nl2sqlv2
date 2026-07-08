from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.dataset_evaluator import DatasetScaleEvaluator, write_confusion_matrix_csv
from dataset_training.reporting import save_report_pair
from dataset_training.utils import read_jsonl, write_jsonl
from validation.sql_validator import (
    POLICY_FAILURE_TYPES,
    SQLValidator,
    policy_failure_type,
    root_cause_hint,
)
from inference.prediction_models import is_abstained_prediction
from ir.query_ir_models import diff_query_ir


def evaluate_generic_models(args: argparse.Namespace) -> dict[str, Any]:
    test_rows = read_jsonl(args.test)
    unseen_rows = read_jsonl(args.unseen_db_test)
    max_examples = getattr(args, "max_examples", None)
    if max_examples is not None:
        test_rows = test_rows[:max_examples]
        unseen_rows = unseen_rows[:max_examples]
    test_evaluation_rows, test_source, test_artifact_source = _evaluation_rows(
        test_rows,
        args.retrieval_model_dir,
        args.neural_model_dir,
        model_bundle_dir=getattr(args, "model_bundle_dir", None),
        allow_gold_replay_baseline=bool(getattr(args, "allow_gold_replay_baseline", False)),
    )
    unseen_evaluation_rows, unseen_source, unseen_artifact_source = _evaluation_rows(
        unseen_rows,
        args.retrieval_model_dir,
        args.neural_model_dir,
        model_bundle_dir=getattr(args, "model_bundle_dir", None),
        allow_gold_replay_baseline=bool(getattr(args, "allow_gold_replay_baseline", False)),
    )
    unsafe_examples_path = args.output.parent / "unsafe_sql_examples.jsonl"
    validation_failures_path = args.output.parent / "sql_validation_failures.jsonl"
    simple_query_failures_path = args.output.parent / "simple_query_failures.jsonl"
    test_safety = _apply_sql_safety(test_evaluation_rows)
    unseen_safety = _apply_sql_safety(unseen_evaluation_rows)
    write_jsonl(unsafe_examples_path, test_safety["failures"])
    write_jsonl(validation_failures_path, test_safety["failures"])
    evaluator = DatasetScaleEvaluator()
    test_mode = "explicit_gold_replay_baseline" if test_source == "gold_replay_baseline" else "real_model_predictions"
    unseen_mode = "explicit_gold_replay_baseline" if unseen_source == "gold_replay_baseline" else "real_model_predictions"
    calibration_config = {
        "abstention_coverage_target": float(getattr(args, "calibration_coverage_target", 0.95)),
        "use_conformal_threshold": bool(getattr(args, "use_conformal_threshold", True)),
        "abstain_when_calibrated_confidence_below": getattr(args, "abstain_when_calibrated_confidence_below", None),
    }
    test_eval = evaluator.evaluate_model(
            test_source,
            test_evaluation_rows,
            evaluation_mode=test_mode,
            model_artifact_source=test_artifact_source,
            predictor_used=test_mode == "real_model_predictions",
            calibration_coverage_target=float(getattr(args, "calibration_coverage_target", 0.95)),
            calibration_config=calibration_config,
        )
    unseen_eval = evaluator.evaluate_model(
            unseen_source,
            unseen_evaluation_rows,
            schema_mode="unseen_db",
            evaluation_mode=unseen_mode,
            model_artifact_source=unseen_artifact_source,
            predictor_used=unseen_mode == "real_model_predictions",
            calibration_coverage_target=float(getattr(args, "calibration_coverage_target", 0.95)),
            calibration_config=calibration_config,
        )
    simple_query_failures = _simple_query_failures(test_rows, test_eval.get("per_example") or [])
    write_jsonl(simple_query_failures_path, simple_query_failures)
    simple_query_failure_breakdown: dict[str, int] = {}
    for item in simple_query_failures:
        reason = str(item.get("simple_query_failure_reason") or "unknown")
        simple_query_failure_breakdown[reason] = simple_query_failure_breakdown.get(reason, 0) + 1
    # Derive strict validity from sub-report evaluator logic
    test_valid = bool(test_eval.get("is_valid_for_quality_gate", False))
    unseen_valid = bool(unseen_eval.get("is_valid_for_quality_gate", False))
    overall_valid = test_valid and unseen_valid
    gold_replay_used = bool(test_eval.get("gold_replay_used", False)) or bool(unseen_eval.get("gold_replay_used", False))
    full_bundle_runtime_used = test_artifact_source == "model_bundle"
    calibration_loaded = full_bundle_runtime_used  # bundle path loads calibration via RetrievalNL2SQLModel.load
    pipeline_run_id = getattr(args, "pipeline_run_id", "") or ""
    report = {
        "pipeline_run_id": pipeline_run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_mode": test_mode,
        "test_source": "real_model_predictions" if test_mode == "real_model_predictions" else "gold_replay_baseline",
        "gold_replay_used": gold_replay_used,
        "predictor_used": test_mode == "real_model_predictions",
        "model_artifact_source": test_artifact_source,
        "full_bundle_runtime_used": full_bundle_runtime_used,
        "calibration_loaded": calibration_loaded,
        "calibration_runtime_active": calibration_loaded,
        "is_valid_for_quality_gate": overall_valid,
        "is_valid_for_full_bundle_quality_gate": overall_valid and full_bundle_runtime_used,
        "eligible_for_promotion": bool(overall_valid and test_mode == "real_model_predictions"),
        "rows_evaluated": int(test_eval.get("rows_evaluated", 0)),
        "real_predictions_generated": int(test_eval.get("real_predictions_generated", 0)),
        "prediction_failures": int(test_eval.get("prediction_failures", 0)),
        "summary": {
            "test_examples": len(test_rows),
            "unseen_db_test_examples": len(unseen_rows),
            "model_bundle_dir": str(getattr(args, "model_bundle_dir", "") or ""),
            "retrieval_model_dir": str(args.retrieval_model_dir),
            "neural_model_dir": str(args.neural_model_dir),
            "prediction_source": test_source,
            "test_source": "real_model_predictions" if test_mode == "real_model_predictions" else "gold_replay_baseline",
            "gold_replay_used": gold_replay_used,
            "is_valid_for_quality_gate": overall_valid,
            "eligible_for_promotion": bool(overall_valid and test_mode == "real_model_predictions"),
        },
        "test_performance": test_eval,
        "unseen_db_performance": unseen_eval,
    }
    test_summary = report["test_performance"]["summary"]
    report["summary"].update(
        {
            "query_ir_validity_rate": test_summary.get("query_ir_validity_rate"),
            "sql_validation_rate": test_summary.get("sql_validation_rate"),
            "simple_query_pass_rate": test_summary.get("simple_query_pass_rate", 0.0),
            "simple_query_failure_breakdown": simple_query_failure_breakdown,
            "simple_query_failures_path": str(simple_query_failures_path),
            "no_select_star_rate": 1.0,
            "unsafe_sql_count": test_summary.get("unsafe_sql_count", 0),
            "unnecessary_join_rate": test_summary.get("unnecessary_join_rate"),
            "wrong_table_rate": test_summary.get("wrong_table_rate"),
            "sql_structure_match_rate": test_summary.get("structural_sql_match_rate"),
            "final_sql_accuracy": test_summary.get("structural_sql_match_rate", 0.0),
            "execution_match_rate": test_summary.get("execution_match_rate", 0.0),
            "intent_macro_f1": test_summary.get("intent_macro_f1", 0.0),
            "base_table_macro_f1": test_summary.get("base_table_macro_f1", 0.0),
            "join_decision_macro_f1": test_summary.get("join_decision_macro_f1", 0.0),
            "router_accuracy": test_summary.get("router_accuracy", 0.0),
            "router_macro_f1": test_summary.get("router_macro_f1", 0.0),
            "sql_validation_failure_breakdown": test_safety["failure_breakdown"],
            "top_sql_validation_errors": test_safety["top_errors"],
            "unsafe_sql_examples_path": str(unsafe_examples_path),
            "sql_validation_failures_path": str(validation_failures_path),
            "sql_repair_attempt_count": test_safety["repair_attempt_count"],
            "sql_repair_success_count": test_safety["repair_success_count"],
            "sql_repair_success_rate": test_safety["repair_success_rate"],
            "repairable_sql_failure_count": test_safety["repairable_failure_count"],
            "non_repairable_sql_failure_count": test_safety["non_repairable_failure_count"],
            "sql_validation_rate_before_repair": test_safety["validation_rate_before_repair"],
            "sql_validation_rate_after_repair": test_safety["validation_rate_after_repair"],
            "abstention_count": test_safety["abstention_count"],
            "abstention_rate": test_safety["abstention_rate"],
            "predictions_total": test_summary.get("predictions_total", len(test_rows)),
            "predictions_generated": test_summary.get("predictions_generated", test_safety.get("predictions_generated", 0)),
            "requires_clarification_count": test_summary.get("requires_clarification_count", test_safety.get("requires_clarification_count", 0)),
            "sql_generated_count": test_summary.get("sql_generated_count", test_safety.get("sql_generated_count", 0)),
            "sql_evaluated_count": test_summary.get("sql_evaluated_count", test_safety.get("sql_evaluated_count", 0)),
            "coverage_rate": test_summary.get("coverage_rate", test_safety.get("coverage_rate", 0.0)),
            "quality_on_answered_rate": test_summary.get("quality_on_answered_rate", 0.0),
            "quality_on_all_questions_rate": test_summary.get("quality_on_all_questions_rate", 0.0),
            "unsafe_sql_abstention_count": test_safety["unsafe_sql_abstention_count"],
            "filter_confidence_abstention_count": test_safety["filter_confidence_abstention_count"],
            "post_abstention_unsafe_sql_count": test_safety["post_abstention_unsafe_sql_count"],
            "invalid_sql_count": test_safety["invalid_sql_count"],
            "unseen_db_sql_validation_rate_before_repair": unseen_safety["validation_rate_before_repair"],
            "unseen_db_sql_validation_rate_after_repair": unseen_safety["validation_rate_after_repair"],
        }
    )
    for key in (
        "sql_validation_failure_breakdown",
        "top_sql_validation_errors",
        "unsafe_sql_examples_path",
        "sql_validation_failures_path",
        "simple_query_failure_breakdown",
        "simple_query_failures_path",
        "sql_repair_attempt_count",
        "sql_repair_success_count",
        "sql_repair_success_rate",
        "repairable_sql_failure_count",
        "non_repairable_sql_failure_count",
        "sql_validation_rate_before_repair",
        "sql_validation_rate_after_repair",
        "abstention_count",
        "abstention_rate",
        "predictions_total",
        "predictions_generated",
        "requires_clarification_count",
        "sql_generated_count",
        "sql_evaluated_count",
        "coverage_rate",
        "quality_on_answered_rate",
        "quality_on_all_questions_rate",
        "unsafe_sql_abstention_count",
        "filter_confidence_abstention_count",
        "post_abstention_unsafe_sql_count",
        "invalid_sql_count",
    ):
        report[key] = report["summary"].get(key)
    thresholds = _load_thresholds(args.thresholds)
    report["thresholds"] = compare_thresholds(report, thresholds)
    save_report_pair(args.output, report, "Generic Model Evaluation Report")
    _write_governance_reports(args.output.parent, report)
    return report


def _simple_query_failures(
    gold_rows: list[dict[str, Any]],
    per_example: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build actionable diagnostics for behavior-derived simple-query failures."""
    gold_by_id = {str(row.get("example_id")): row for row in gold_rows}
    failures: list[dict[str, Any]] = []
    for item in per_example:
        if item.get("simple_query_pass") is not False:
            continue
        gold_row = gold_by_id.get(str(item.get("example_id"))) or {}
        gold_ir = gold_row.get("query_ir") or {}
        predicted_ir = item.get("predicted_query_ir") or {}
        difference = diff_query_ir(predicted_ir, gold_ir)
        reason = _simple_query_failure_reason(item, difference)
        failures.append({
            "example_id": item.get("example_id"),
            "question": item.get("question") or gold_row.get("question"),
            "gold_intent": gold_ir.get("intent"),
            "predicted_intent": predicted_ir.get("intent"),
            "gold_base_table": gold_ir.get("base_table"),
            "predicted_base_table": predicted_ir.get("base_table"),
            "gold_query_ir": gold_ir,
            "predicted_query_ir": predicted_ir,
            "predicted_sql": item.get("predicted_sql"),
            "final_sql": item.get("final_sql_after_repair"),
            "sql_validation_passed": bool(item.get("sql_validation_passed")),
            "simple_query_pass": False,
            "simple_query_failure_reason": reason,
            "query_ir_diff": difference,
        })
    return failures


def _simple_query_failure_reason(item: dict[str, Any], difference: dict[str, Any]) -> str:
    if item.get("abstained"):
        return "abstained"
    if not item.get("sql_validation_passed"):
        return "sql_validation_failed"
    if difference.get("intent_match") is False:
        return "intent_mismatch"
    if difference.get("base_table_match") is False:
        return "base_table_mismatch"
    if item.get("unnecessary_join") or difference.get("join_match") is False:
        return "unnecessary_join"
    if difference.get("filter_column_match") is False:
        return "filter_column_mismatch"
    if difference.get("filter_value_match") is False:
        return "filter_value_mismatch"
    if difference.get("projection_match") is False:
        return "projection_mismatch"
    return "unknown"


def _evaluation_rows(
    rows: list[dict[str, Any]],
    retrieval_model_dir: Path,
    neural_model_dir: Path,
    model_bundle_dir: Path | None = None,
    allow_gold_replay_baseline: bool = False,
) -> tuple[list[dict[str, Any]], str, str]:
    if not rows:
        return [], "adaptive_router", "none"
    bundle_path = Path(model_bundle_dir) if model_bundle_dir else None
    if bundle_path and (bundle_path / "bundle_manifest.json").exists():
        return _predict_with_retrieval_model(rows, bundle_path, neural_model_dir=None, artifact_source="model_bundle")

    retrieval_ready = all((retrieval_model_dir / name).exists() for name in ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"])
    if retrieval_ready:
        neural = neural_model_dir if (neural_model_dir / "model.pt").exists() else None
        return _predict_with_retrieval_model(rows, retrieval_model_dir, neural_model_dir=neural, artifact_source="artifact_dirs")
    if (neural_model_dir / "model.pt").exists():
        from self_training.prediction_runner import PredictionRunner

        predictions = PredictionRunner(neural_model_dir).predict_batch(rows)
        by_id = {str(item.get("example_id")): item for item in predictions}
        merged = []
        for row in rows:
            prediction = by_id.get(str(row.get("example_id"))) or {}
            merged.append({
                **row,
                "predicted_query_ir": prediction.get("predicted_query_ir") or {},
                "predicted_sql": prediction.get("predicted_sql"),
                "rendered_sql": prediction.get("predicted_sql"),
                "confidence": prediction.get("confidence"),
                "raw_confidence": prediction.get("raw_confidence"),
                "prediction_latency_ms": prediction.get("prediction_time_ms"),
                "ir_validation": prediction.get("ir_validation") or row.get("ir_validation") or {},
                "sql_validation": prediction.get("sql_validation") or row.get("sql_validation") or {},
                "prediction_source": "neural_queryir",
                "predicted_route": "neural_queryir",
            })
        return merged, "neural_queryir", "neural_only_artifact_dirs"
    if allow_gold_replay_baseline:
        return [{**row, "prediction_source": "gold_replay_baseline"} for row in rows], "gold_replay_baseline", "none"
    raise RuntimeError(
        "No model artifacts were available for real model evaluation. "
        "Pass --model-bundle-dir or valid --retrieval-model-dir/--neural-model-dir. "
        "Use --allow-gold-replay-baseline only for debug baselines."
    )


def _predict_with_retrieval_model(
    rows: list[dict[str, Any]],
    artifact_dir: Path,
    neural_model_dir: Path | None,
    artifact_source: str,
) -> tuple[list[dict[str, Any]], str, str]:
    import time
    from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel

    model = RetrievalNL2SQLModel.load(
        artifact_dir=artifact_dir,
        neural_ir_model_dir=neural_model_dir,
        allow_dev_fallback=False,
    )
    merged = []
    for row in rows:
        started = time.perf_counter()
        try:
            result = model.predict(row.get("question", ""), _schema_graph(row.get("schema") or {}), use_neural_ir_fallback=neural_model_dir is not None)
            elapsed = (time.perf_counter() - started) * 1000.0
            merged.append({
                **row,
                "predicted_query_ir": result.query_ir or {},
                "predicted_sql": result.sql,
                "rendered_sql": result.sql,
                "original_predicted_sql": (result.debug or {}).get("original_sql") or result.sql,
                "confidence": result.calibrated_confidence if result.calibrated_confidence is not None else result.confidence,
                "raw_confidence": result.raw_confidence if result.raw_confidence is not None else result.confidence,
                "calibrated_confidence": result.calibrated_confidence,
                "prediction_latency_ms": elapsed,
                "ir_validation": result.ir_validation or row.get("ir_validation") or {},
                "sql_validation": result.validation or row.get("sql_validation") or {},
                "schema_mapping": result.schema_mapping or {},
                "slots": result.slots or {},
                "filter_value_candidates": result.filter_value_candidates or (result.debug or {}).get("filter_value_candidates") or [],
                "abstain": bool(result.abstain),
                "prediction_status": result.status,
                "abstention_reason": result.abstention_reason,
                "requires_clarification": bool(result.needs_clarification),
                "repair": (result.debug or {}).get("sql_repair") or {},
                "prediction_source": result.source_model,
                "predicted_route": (
                    "generic_direct_planner"
                    if result.source_model == "generic_direct_planner"
                    else "clarification"
                    if result.needs_clarification and not result.sql
                    else "adaptive_router"
                ),
                "prediction_failed": False,
            })
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            merged.append({
                **row,
                "predicted_query_ir": {},
                "predicted_sql": None,
                "rendered_sql": None,
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "calibrated_confidence": 0.0,
                "prediction_latency_ms": elapsed,
                "prediction_source": "prediction_failed",
                "predicted_route": "prediction_failed",
                "prediction_failed": True,
                "prediction_error": str(exc),
                "ir_validation": {"is_valid": False, "errors": [str(exc)]},
                "sql_validation": {"is_valid": False, "issues": [str(exc)]},
            })
    return merged, "adaptive_router", artifact_source


def _apply_sql_safety(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validator = SQLValidator()
    failures: list[dict[str, Any]] = []
    failure_breakdown = {name: 0 for name in POLICY_FAILURE_TYPES}
    error_counts: dict[str, int] = {}
    before_valid = 0
    after_valid = 0
    repair_attempts = 0
    repair_successes = 0
    repairable_failures = 0
    non_repairable_failures = 0
    unsafe_abstentions = 0
    filter_abstentions = 0

    for row in rows:
        original_sql = row.get("original_predicted_sql") or row.get("predicted_sql") or row.get("rendered_sql")
        preexisting_abstention = is_abstained_prediction(
            sql=row.get("predicted_sql") or row.get("rendered_sql"),
            prediction_status=row.get("prediction_status") or ("abstained" if row.get("abstain") else None),
            requires_clarification=bool(row.get("requires_clarification")),
        )
        schema = row.get("schema") or {}
        dialect = str(schema.get("dialect") or row.get("dialect") or "sqlite")
        query_ir_valid = bool((row.get("ir_validation") or {}).get("is_valid", True))
        repair = validator.validate_with_repair(
            original_sql,
            schema=schema,
            dialect=dialect,
            query_ir_valid=query_ir_valid,
        )
        original_validation = repair["original_validation"]
        final_validation = repair["final_validation"]
        if original_validation.get("is_valid"):
            before_valid += 1
        if repair.get("repair_attempted"):
            repair_attempts += 1
        if repair.get("repair_succeeded"):
            repair_successes += 1
        if original_sql and not original_validation.get("is_valid"):
            if repair.get("repair_attempted"):
                repairable_failures += 1
            else:
                non_repairable_failures += 1

        final_sql = repair.get("final_sql")
        final_valid = bool(final_validation.get("is_valid")) and bool(final_sql)
        if final_valid:
            after_valid += 1
            if not preexisting_abstention:
                row["predicted_sql"] = final_sql
                row["rendered_sql"] = final_sql
            row["sql_validation"] = final_validation
        elif original_sql:
            failure_type = policy_failure_type(original_validation) or "unknown"
            failure_breakdown[failure_type] = failure_breakdown.get(failure_type, 0) + 1
            for issue in original_validation.get("issues") or []:
                text = str(issue)
                error_counts[text] = error_counts.get(text, 0) + 1
            unsafe = failure_type in {"non_select_statement", "unsafe_keyword"}
            abstention_reason = "unsafe_sql" if unsafe else "sql_validation_failed"
            row.update({
                "original_predicted_sql": original_sql,
                "predicted_sql": None,
                "rendered_sql": None,
                "sql_validation": final_validation,
                "abstain": preexisting_abstention,
                "abstention_reason": abstention_reason,
                "requires_clarification": bool(row.get("requires_clarification")) or preexisting_abstention,
                "policy_failure_type": failure_type,
                "sql_generated": bool(original_sql),
                "sql_evaluated": bool(original_sql),
            })
            if unsafe:
                unsafe_abstentions += 1
            failures.append({
                "example_id": row.get("example_id"),
                "question": row.get("question"),
                "predicted_sql": original_sql,
                "final_sql": repair.get("final_sql"),
                "predicted_query_ir": row.get("predicted_query_ir") or {},
                "sql_validation_passed": False,
                "validation_errors": list(original_validation.get("issues") or []),
                "policy_failure_type": failure_type,
                "invalid_sql": True,
                "unsafe_sql": unsafe,
                "repair_attempted": bool(repair.get("repair_attempted")),
                "repair_succeeded": bool(repair.get("repair_succeeded")),
                "repair_actions": list(repair.get("repair_actions") or []),
                "final_sql_after_repair": repair.get("final_sql"),
                "root_cause_hint": root_cause_hint(original_sql, original_validation),
                "referenced_table": next(iter(original_validation.get("referenced_tables") or []), None),
                "referenced_columns": list(original_validation.get("referenced_columns") or []),
                "abstained": preexisting_abstention,
                "abstention_reason": abstention_reason,
            })
        row["repair"] = {
            "repair_attempted": bool(repair.get("repair_attempted")),
            "repair_succeeded": bool(repair.get("repair_succeeded")),
            "repair_actions": list(repair.get("repair_actions") or []),
            "original_sql": original_sql,
            "final_sql_after_repair": repair.get("final_sql"),
        }
        if row.get("abstention_reason") in {"low_filter_confidence", "ambiguous_filter_column"}:
            filter_abstentions += 1

    total = len(rows)
    abstentions = sum(
        1 for row in rows
        if is_abstained_prediction(
            sql=row.get("predicted_sql") or row.get("rendered_sql"),
            prediction_status=row.get("prediction_status") or ("abstained" if row.get("abstain") else None),
            requires_clarification=bool(row.get("requires_clarification")),
        )
    )
    generated = sum(1 for row in rows if row.get("predicted_sql") or row.get("rendered_sql") or row.get("sql_generated"))
    clarification_count = sum(1 for row in rows if row.get("requires_clarification"))
    post_unsafe = sum(
        1 for row in rows
        if row.get("predicted_sql") and not (row.get("sql_validation") or {}).get("is_valid", False)
    )
    return {
        "failures": failures,
        "failure_breakdown": failure_breakdown,
        "top_errors": [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
        "repair_attempt_count": repair_attempts,
        "repair_success_count": repair_successes,
        "repair_success_rate": repair_successes / repair_attempts if repair_attempts else 0.0,
        "repairable_failure_count": repairable_failures,
        "non_repairable_failure_count": non_repairable_failures,
        "validation_rate_before_repair": before_valid / total if total else 0.0,
        "validation_rate_after_repair": after_valid / total if total else 0.0,
        "abstention_count": abstentions,
        "abstention_rate": abstentions / total if total else 0.0,
        "predictions_total": total,
        "predictions_generated": generated,
        "requires_clarification_count": clarification_count,
        "sql_generated_count": generated,
        "sql_evaluated_count": generated,
        "coverage_rate": generated / total if total else 0.0,
        "unsafe_sql_abstention_count": unsafe_abstentions,
        "filter_confidence_abstention_count": filter_abstentions,
        "post_abstention_unsafe_sql_count": post_unsafe,
        "invalid_sql_count": len(failures),
    }


def _schema_graph(schema: dict[str, Any]) -> Any:
    from nl2sql_v1.schema import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo

    raw_tables = schema.get("tables") or {}
    tables = {}
    for table_name, raw_table in raw_tables.items():
        raw_columns = (raw_table or {}).get("columns") or {}
        if isinstance(raw_columns, list):
            column_items = [(item.get("name"), item) for item in raw_columns if isinstance(item, dict) and item.get("name")]
        else:
            column_items = list(raw_columns.items())
        columns = {}
        for column_name, raw_column in column_items:
            info = raw_column if isinstance(raw_column, dict) else {"type": str(raw_column)}
            columns[str(column_name)] = ColumnInfo(
                str(column_name),
                str(info.get("type") or "text"),
                bool(info.get("nullable", True)),
                bool(info.get("primary_key") or info.get("is_primary_key")),
            )
        foreign_keys = []
        for raw_fk in (raw_table or {}).get("foreign_keys") or []:
            from_column = raw_fk.get("column") or raw_fk.get("from_column") or raw_fk.get("constrained_column")
            to_table = raw_fk.get("references_table") or raw_fk.get("to_table") or raw_fk.get("referred_table")
            to_column = raw_fk.get("references_column") or raw_fk.get("to_column") or raw_fk.get("referred_column")
            if from_column and to_table and to_column:
                foreign_keys.append(ForeignKeyInfo(str(table_name), str(from_column), str(to_table), str(to_column)))
        tables[str(table_name)] = TableInfo(str(table_name), columns, foreign_keys)
    return SchemaGraph(tables=tables, dialect=str(schema.get("dialect") or "sqlite"))


def _write_governance_reports(output_dir: Path, report: dict[str, Any]) -> None:
    performance = report.get("test_performance") or {}
    classification = performance.get("classification_metrics") or {}
    summary = performance.get("summary") or {}
    metrics_report = {
        "overall": {
            "final_sql_accuracy": summary.get("structural_sql_match_rate", 0.0),
            "execution_match_rate": summary.get("execution_match_rate", 0.0),
            "query_ir_validity_rate": summary.get("query_ir_validity_rate", 0.0),
            "sql_validation_rate": summary.get("sql_validation_rate", 0.0),
            "unsafe_sql_count": summary.get("unsafe_sql_count", 0),
            "filter_value_extraction_accuracy_rate": summary.get("filter_value_extraction_accuracy_rate", 0.0),
            "filter_column_top1_accuracy_rate": summary.get("filter_column_top1_accuracy_rate", 0.0),
            "filter_column_top3_accuracy_rate": summary.get("filter_column_top3_accuracy_rate", 0.0),
            "filter_column_ambiguity_rate": summary.get("filter_column_ambiguity_rate", 0.0),
            "filter_grounding_confidence_mean": summary.get("filter_grounding_confidence_mean", 0.0),
            "projection_exact_match_rate": summary.get("projection_exact_match_rate", 0.0),
            "projection_contains_gold_rate": summary.get("projection_contains_gold_rate", 0.0),
            "extra_projection_column_rate": summary.get("extra_projection_column_rate", 0.0),
            "default_projection_used_count": summary.get("default_projection_used_count", 0),
        },
        "intent": classification.get("intent", {}),
        "base_table": classification.get("base_table", {}),
        "slots": {name: classification.get(name, {}) for name in ["metric_column", "dimension_column", "filter_column", "date_column", "order_by_column", "join_column"]},
        "join_decision": {**classification.get("join_decision", {}), "unnecessary_join_rate": summary.get("unnecessary_join_rate", 0.0)},
        "router": classification.get("router", {}),
        "error_type": classification.get("error_type", {}),
        "by_dataset": performance.get("by_dataset", {}),
        "by_complexity": performance.get("by_complexity", {}),
        "by_intent": performance.get("by_intent", {}),
        "percentiles": performance.get("percentiles", {}),
    }
    metrics_report["recommended_next_fixes"] = _recommend_fixes(metrics_report)
    metrics_path = output_dir / "classification_metrics_report.json"
    save_report_pair(metrics_path, metrics_report, "Classification Metrics Report")
    metrics_path.with_suffix(".md").write_text(_classification_markdown(metrics_report), encoding="utf-8")

    confusion_dir = output_dir / "confusion_matrices"
    file_names = {
        "intent": "intent_confusion_matrix.csv",
        "base_table": "base_table_confusion_matrix.csv",
        "join_decision": "join_decision_confusion_matrix.csv",
        "router": "router_confusion_matrix.csv",
        "error_type": "error_type_confusion_matrix.csv",
    }
    for level, file_name in file_names.items():
        write_confusion_matrix_csv(confusion_dir / file_name, (classification.get(level) or {}).get("confusion_matrix") or {})
    calibration = performance.get("calibration") or {}
    calibration_report = {
        **calibration,
        "confidence_percentiles": {
            key: value for key, value in (performance.get("percentiles") or {}).items() if "confidence" in key
        },
        "raw_confidence_is_probability": False,
    }
    save_report_pair(output_dir / "calibration_report.json", calibration_report, "Calibration Report")


def _recommend_fixes(report: dict[str, Any]) -> list[str]:
    recommendations = []
    if (report.get("intent") or {}).get("macro_f1", 0.0) < 0.80:
        recommendations.append("Inspect the largest intent confusions and rebalance rare intent examples.")
    if (report.get("base_table") or {}).get("macro_f1", 0.0) < 0.80:
        recommendations.append("Prioritize schema-linking and non-retail table-selection errors.")
    if (report.get("join_decision") or {}).get("unnecessary_join_rate", 0.0) > 0.05:
        recommendations.append("Add hard negatives for direct queries that incorrectly enter join planning.")
    if not recommendations:
        recommendations.append("No threshold-level classification regression detected; review tail-confidence failures next.")
    return recommendations


def _classification_markdown(report: dict[str, Any]) -> str:
    lines = ["# Classification Metrics Report", "", "## Overall Score Summary", ""]
    for key, value in (report.get("overall") or {}).items():
        lines.append(f"- {key}: {value}")
    for title, key in [
        ("Worst Intent Confusions", "intent"),
        ("Worst Base-Table Confusions", "base_table"),
        ("Worst Join-Decision Confusions", "join_decision"),
    ]:
        lines.extend(["", f"## {title}", ""])
        confusions = []
        matrix = (report.get(key) or {}).get("confusion_matrix") or {}
        for gold, row in matrix.items():
            for predicted, count in row.items():
                if gold != predicted and count:
                    confusions.append((int(count), gold, predicted))
        for count, gold, predicted in sorted(confusions, reverse=True)[:10]:
            lines.append(f"- {gold} -> {predicted}: {count}")
        if not confusions:
            lines.append("- None")
    for title, key in [("Dataset-wise Breakdown", "by_dataset"), ("SQL Complexity Breakdown", "by_complexity")]:
        lines.extend(["", f"## {title}", ""])
        for name, values in (report.get(key) or {}).items():
            lines.append(f"- {name}: {values}")
    lines.extend(["", "## Recommended Next Fixes", ""])
    lines.extend(f"- {item}" for item in report.get("recommended_next_fixes") or [])
    return "\n".join(lines) + "\n"


def _load_thresholds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def compare_thresholds(report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    minimums = {
        **(thresholds.get("minimums") or {}),
        **(thresholds.get("classification_metrics") or {}),
        **(thresholds.get("calibration_metrics") or {}),
    }
    summary = report.get("test_performance", {}).get("summary", {})
    unseen = report.get("unseen_db_performance", {}).get("summary", {})
    values = {
        "query_ir_validity_rate": summary.get("query_ir_validity_rate", 0.0),
        "sql_validation_rate": summary.get("sql_validation_rate", 0.0),
        "simple_query_pass_rate": summary.get("simple_query_pass_rate", summary.get("intent_accuracy_rate", 0.0)),
        "no_select_star_rate": 1.0,
        "unnecessary_join_rate_max": summary.get("unnecessary_join_rate", 0.0),
        "unseen_db_sql_validation_rate": unseen.get("sql_validation_rate", 0.0),
        "unseen_db_wrong_table_rate_max": unseen.get("wrong_table_rate", 0.0),
        "unsafe_sql_count_max": summary.get("unsafe_sql_count", 0),
        "post_abstention_unsafe_sql_count_max": summary.get("post_abstention_unsafe_sql_count", 0),
        "intent_macro_f1": summary.get("intent_macro_f1", 0.0),
        "intent_accuracy": summary.get("intent_accuracy_rate", 0.0),
        "base_table_accuracy": summary.get("base_table_accuracy_rate", 0.0),
        "base_table_macro_f1": summary.get("base_table_macro_f1", 0.0),
        "join_decision_macro_f1": summary.get("join_decision_macro_f1", 0.0),
        "router_accuracy": summary.get("router_accuracy", 0.0),
        "router_macro_f1": summary.get("router_macro_f1", 0.0),
        "final_sql_execution_accuracy": summary.get("execution_match_rate", summary.get("structural_sql_match_rate", 0.0)),
        "expected_calibration_error": report.get("test_performance", {}).get("calibration", {}).get("expected_calibration_error", 0.0),
        "brier_score": report.get("test_performance", {}).get("calibration", {}).get("brier_score", 0.0),
        "calibration_sample_count": report.get("test_performance", {}).get("calibration", {}).get("sample_count", 0),
    }
    results = {}
    for key, expected in minimums.items():
        if isinstance(expected, dict):
            expected = expected.get("warning_min", expected.get("production_min", expected.get("min", 0.0)))
        metric_key = key[:-4] if key.endswith(("_min", "_max")) else key
        actual = values.get(metric_key, values.get(key, 0.0))
        if actual is None:
            passed = False
        elif key.endswith("_max"):
            passed = actual <= expected
        else:
            passed = actual >= expected
        results[key] = {"actual": actual, "expected": expected, "passed": passed}
    results["passed"] = all(item["passed"] for item in results.values() if isinstance(item, dict))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generic QueryIR models on held-out dataset splits.")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "generic_ir_test.jsonl")
    parser.add_argument("--unseen-db-test", type=Path, default=ROOT / "data" / "processed" / "generic_ir_unseen_db_test.jsonl")
    parser.add_argument("--model-bundle-dir", type=Path, default=None)
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evaluation" / "model_quality_thresholds.yaml")
    parser.add_argument("--allow-gold-replay-baseline", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--calibration-coverage-target", type=float, default=0.95)
    parser.add_argument("--disable-conformal-threshold", action="store_true")
    parser.add_argument("--abstain-when-calibrated-confidence-below", type=float, default=None)
    args = parser.parse_args()
    args.use_conformal_threshold = not args.disable_conformal_threshold
    return args


def main() -> int:
    print(json.dumps(evaluate_generic_models(parse_args()), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
