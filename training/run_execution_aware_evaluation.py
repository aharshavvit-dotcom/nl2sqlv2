from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl, write_json
from execution_eval.execution_reporter import ExecutionReporter
from execution_eval.sql_structure_comparator import SQLStructureComparator


def _schema(row: dict[str, Any]) -> dict[str, Any]:
    schema = row.get("schema") or row.get("schema_context") or {}
    if isinstance(schema, dict) and schema.get("tables"):
        return schema
    query_ir = row.get("gold_query_ir") or row.get("query_ir") or row.get("predicted_query_ir") or {}
    context = ((query_ir.get("metadata") or {}).get("validation_context") or {}).get("schema_context") or {}
    return {"dialect": query_ir.get("dialect") or row.get("dialect") or "sqlite", "tables": context.get("tables", {})}


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparator = SQLStructureComparator()
    evaluated = []
    for row in rows:
        predicted_sql = row.get("predicted_sql") or row.get("sql") or row.get("generated_sql") or ""
        gold_sql = row.get("gold_sql") or row.get("source_sql") or row.get("rendered_sql") or ""
        dialect = row.get("dialect") or _schema(row).get("dialect", "sqlite")
        structure = comparator.compare(predicted_sql, gold_sql, schema=_schema(row), dialect=dialect)
        evaluated.append(
            {
                "example_id": row.get("example_id"),
                "question": row.get("question"),
                "dataset_name": row.get("dataset_name"),
                "intent": (row.get("gold_query_ir") or row.get("query_ir") or {}).get("intent") or row.get("intent"),
                "execution_available": False,
                "execution_match": None,
                "structure": structure,
                "predicted_sql": predicted_sql,
                "gold_sql": gold_sql,
            }
        )
    return ExecutionReporter().summarize(evaluated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-aware evaluation on prediction rows.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "execution_aware_evaluation_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}. Run training/evaluate_against_gold.py first.")
    rows = read_jsonl(args.predictions)
    report = evaluate_rows(rows)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
