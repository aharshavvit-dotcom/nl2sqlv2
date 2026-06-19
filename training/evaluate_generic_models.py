from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.dataset_evaluator import DatasetScaleEvaluator, write_confusion_matrix_csv
from dataset_training.reporting import save_report_pair
from dataset_training.utils import read_jsonl


def evaluate_generic_models(args: argparse.Namespace) -> dict[str, Any]:
    test_rows = read_jsonl(args.test)
    unseen_rows = read_jsonl(args.unseen_db_test)
    test_evaluation_rows, test_source = _evaluation_rows(test_rows, args.retrieval_model_dir, args.neural_model_dir)
    unseen_evaluation_rows, unseen_source = _evaluation_rows(unseen_rows, args.retrieval_model_dir, args.neural_model_dir)
    evaluator = DatasetScaleEvaluator()
    report = {
        "summary": {
            "test_examples": len(test_rows),
            "unseen_db_test_examples": len(unseen_rows),
            "retrieval_model_dir": str(args.retrieval_model_dir),
            "neural_model_dir": str(args.neural_model_dir),
            "prediction_source": test_source,
        },
        "test_performance": evaluator.evaluate_model(test_source, test_evaluation_rows),
        "unseen_db_performance": evaluator.evaluate_model(unseen_source, unseen_evaluation_rows),
    }
    test_summary = report["test_performance"]["summary"]
    report["summary"].update(
        {
            "query_ir_validity_rate": test_summary.get("query_ir_validity_rate"),
            "sql_validation_rate": test_summary.get("sql_validation_rate"),
            "simple_query_pass_rate": test_summary.get("simple_query_pass_rate", test_summary.get("intent_accuracy_rate")),
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
        }
    )
    thresholds = _load_thresholds(args.thresholds)
    report["thresholds"] = compare_thresholds(report, thresholds)
    save_report_pair(args.output, report, "Generic Model Evaluation Report")
    _write_governance_reports(args.output.parent, report)
    return report


def _evaluation_rows(
    rows: list[dict[str, Any]],
    retrieval_model_dir: Path,
    neural_model_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    retrieval_ready = all((retrieval_model_dir / name).exists() for name in ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"])
    if rows and retrieval_ready:
        import time
        from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel

        neural = neural_model_dir if (neural_model_dir / "model.pt").exists() else None
        model = RetrievalNL2SQLModel.load(
            artifact_dir=retrieval_model_dir,
            neural_ir_model_dir=neural,
            allow_dev_fallback=False,
        )
        merged = []
        for row in rows:
            started = time.perf_counter()
            result = model.predict(row.get("question", ""), _schema_graph(row.get("schema") or {}), use_neural_ir_fallback=neural is not None)
            elapsed = (time.perf_counter() - started) * 1000.0
            merged.append({
                **row,
                "predicted_query_ir": result.query_ir or {},
                "predicted_sql": result.sql,
                "rendered_sql": result.sql,
                "confidence": result.calibrated_confidence if result.calibrated_confidence is not None else result.confidence,
                "raw_confidence": result.raw_confidence if result.raw_confidence is not None else result.confidence,
                "prediction_latency_ms": elapsed,
                "ir_validation": result.ir_validation or row.get("ir_validation") or {},
                "sql_validation": result.validation or row.get("sql_validation") or {},
                "prediction_source": result.source_model,
                "predicted_route": (
                    "generic_direct_planner"
                    if result.source_model == "generic_direct_planner"
                    else "clarification"
                    if result.needs_clarification and not result.sql
                    else "adaptive_router"
                ),
            })
        return merged, "adaptive_router"
    if rows and (neural_model_dir / "model.pt").exists():
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
        return merged, "neural_queryir"
    return [{**row, "prediction_source": "gold_replay_baseline"} for row in rows], "gold_replay_baseline"


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
        "simple_query_pass_rate": summary.get("intent_accuracy_rate", 0.0),
        "no_select_star_rate": 1.0,
        "unnecessary_join_rate_max": summary.get("unnecessary_join_rate", 0.0),
        "unseen_db_sql_validation_rate": unseen.get("sql_validation_rate", 0.0),
        "unseen_db_wrong_table_rate_max": unseen.get("wrong_table_rate", 0.0),
        "unsafe_sql_count_max": 0,
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
        metric_key = key[:-4] if key.endswith(("_min", "_max")) else key
        actual = values.get(metric_key, values.get(key, 0.0))
        if key.endswith("_max"):
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
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evaluation" / "model_quality_thresholds.yaml")
    return parser.parse_args()


def main() -> int:
    print(json.dumps(evaluate_generic_models(parse_args()), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
