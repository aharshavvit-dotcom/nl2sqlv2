from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.query_executor import execute_select
from neural_ir.evaluator import OptionAIREvaluator
from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch
from neural_ir.model_registry import load_model_bundle
from neural_ir.predictor import OptionAIRPredictor
from nl2sql_v1.schema import read_sqlite_schema


def evaluate_option_a_v2_model(
    model_dir: Path,
    test_path: Path,
    output_path: Path,
    eval_cases_path: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    bundle = load_model_bundle(model_dir)
    config = bundle["config"]
    rows = _load_jsonl(test_path)
    if rows:
        dataset = IRTrainingDataset(
            str(test_path),
            vocab=bundle["vocab"],
            label_encoder=bundle["label_encoder"],
            max_question_len=int(config.get("max_question_len", 64)),
            max_schema_len=int(config.get("max_schema_len", 320)),
            max_candidate_tokens=int(config.get("max_candidate_tokens", 16)),
            max_tables=int(config.get("max_tables", 64)),
            max_columns=int(config.get("max_columns", 256)),
        )
        loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 8)), shuffle=False, collate_fn=collate_ir_batch)
        base = OptionAIREvaluator().evaluate(bundle["model"], loader, bundle["label_encoder"], db_path=str(db_path) if db_path else None)
    else:
        base = {"total_examples": 0, "by_intent": {}, "sample_failures": []}
    case_report = evaluate_cases(model_dir, eval_cases_path, db_path) if eval_cases_path else {"end_to_end_case_pass_rate": 0.0, "execution_success_rate": 0.0, "eval_cases": []}
    summary = _summary(base, case_report)
    report = {
        "summary": summary,
        "by_intent": base.get("by_intent", {}),
        "by_dataset": _by_field(rows, "dataset_name"),
        "by_difficulty": _by_field(rows, "difficulty"),
        "repair_stats": case_report.get("repair_stats", {}),
        "sample_failures": base.get("sample_failures", [])[:25],
        "recommendations": _recommendations(summary),
        "eval_cases": case_report.get("eval_cases", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def evaluate_cases(model_dir: Path, eval_cases_path: Path | None, db_path: Path | None) -> dict[str, Any]:
    cases = _load_jsonl(eval_cases_path) if eval_cases_path else []
    if not cases:
        return {"end_to_end_case_pass_rate": 0.0, "execution_success_rate": 0.0, "eval_cases": [], "repair_stats": {}}
    if db_path is None:
        return {
            "end_to_end_case_pass_rate": 0.0,
            "execution_success_rate": 0.0,
            "eval_cases": [{"id": row.get("id"), "passed": False, "reason": "db not provided"} for row in cases],
            "repair_stats": {},
        }
    schema = read_sqlite_schema(db_path)
    predictor = OptionAIRPredictor(str(model_dir))
    rows = []
    passed = 0
    executed = 0
    repaired = 0
    for row in cases:
        result = predictor.predict(row["question"], schema)
        sql = result.get("sql") or ""
        sql_upper = sql.upper()
        contains_ok = all(str(fragment).upper() in sql_upper for fragment in row.get("expected_sql_contains", []))
        not_contains_ok = all(str(fragment).upper() not in sql_upper for fragment in row.get("expected_sql_not_contains", []))
        valid = bool((result.get("sql_validation") or {}).get("is_valid"))
        if result.get("repairs_applied"):
            repaired += 1
        execution_ok = False
        if valid and sql and row.get("should_execute", True):
            try:
                execute_select(db_path, sql, validation_result=result.get("sql_validation"))
                execution_ok = True
                executed += 1
            except Exception:
                execution_ok = False
        case_passed = valid and contains_ok and not_contains_ok and (execution_ok or not row.get("should_execute", True))
        if case_passed:
            passed += 1
        rows.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "expected_intent": row.get("expected_intent"),
                "predicted_intent": (result.get("debug") or {}).get("decoded_prediction", {}).get("intent"),
                "sql_valid": valid,
                "execution_success": execution_ok,
                "contains_ok": contains_ok,
                "not_contains_ok": not_contains_ok,
                "repairs_applied": result.get("repairs_applied", []),
                "passed": case_passed,
                "sql": sql,
            }
        )
    return {
        "end_to_end_case_pass_rate": passed / max(len(cases), 1),
        "execution_success_rate": executed / max(len(cases), 1),
        "eval_cases": rows,
        "repair_stats": {"cases_with_repairs": repaired, "repair_success_rate": repaired / max(len(cases), 1)},
    }


def _summary(base: dict[str, Any], case_report: dict[str, Any]) -> dict[str, Any]:
    sql_rate = float(base.get("sql_validation_rate") or 0.0)
    ir_rate = float(base.get("query_ir_validity_rate") or 0.0)
    return {
        "intent_accuracy": float(base.get("intent_accuracy") or 0.0),
        "base_table_accuracy": float(base.get("base_table_accuracy") or 0.0),
        "metric_column_accuracy": float(base.get("metric_column_accuracy") or 0.0),
        "dimension_column_accuracy": float(base.get("dimension_column_accuracy") or 0.0),
        "date_column_accuracy": float(base.get("date_column_accuracy") or 0.0),
        "filter_column_accuracy": float(base.get("filter_column_accuracy") or 0.0),
        "metric_expression_type_accuracy": float(base.get("metric_expression_type_accuracy") or 0.0),
        "date_grain_accuracy": float(base.get("date_grain_accuracy") or 0.0),
        "filter_operator_accuracy": float(base.get("filter_operator_accuracy") or 0.0),
        "limit_bucket_accuracy": float(base.get("limit_bucket_accuracy") or 0.0),
        "query_ir_validity_rate_before_repair": ir_rate,
        "query_ir_validity_rate_after_repair": ir_rate,
        "sql_validation_rate_before_repair": sql_rate,
        "sql_validation_rate_after_repair": sql_rate,
        "repair_success_rate": float((case_report.get("repair_stats") or {}).get("repair_success_rate") or 0.0),
        "execution_success_rate": float(case_report.get("execution_success_rate") or 0.0),
        "end_to_end_case_pass_rate": float(case_report.get("end_to_end_case_pass_rate") or 0.0),
    }


def _by_field(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts.setdefault(key, {"total": 0})
        counts[key]["total"] += 1
    return counts


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recommendations = []
    if summary["metric_column_accuracy"] < 0.8:
        recommendations.append("add more hard negatives for similar metric columns")
    if summary["end_to_end_case_pass_rate"] < 0.8:
        recommendations.append("inspect sample failures and add targeted repair/eval cases")
    return recommendations


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Option A V2 QueryIR model.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--eval-cases", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_option_a_v2_model(args.model_dir, args.test, args.output, args.eval_cases, args.db)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
