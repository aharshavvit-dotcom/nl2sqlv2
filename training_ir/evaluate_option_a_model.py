from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from execution.query_executor import execute_select
from neural_ir.evaluator import OptionAIREvaluator
from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch
from neural_ir.model_registry import load_model_bundle
from neural_ir.predictor import OptionAIRPredictor
from nl2sql_v1.schema import read_sqlite_schema


DEFAULT_MODEL_DIR = ROOT / "artifacts" / "option_a_ir_model"


def evaluate_option_a_model(
    model_dir: Path,
    test_path: Path,
    output_path: Path,
    eval_cases_path: Path | None = None,
    db_path: Path | None = None,
) -> dict:
    bundle = load_model_bundle(model_dir)
    config = bundle["config"]
    dataset = IRTrainingDataset(
        str(test_path),
        vocab=bundle["vocab"],
        label_encoder=bundle["label_encoder"],
        max_question_len=int(config.get("max_question_len", 64)),
        max_schema_len=int(config.get("max_schema_len", 256)),
        max_tables=int(config.get("max_tables", 64)),
        max_columns=int(config.get("max_columns", 256)),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 16)), shuffle=False, collate_fn=collate_ir_batch)
    report = OptionAIREvaluator().evaluate(bundle["model"], loader, bundle["label_encoder"], db_path=str(db_path) if db_path else None)
    if eval_cases_path:
        case_report = evaluate_cases(model_dir, eval_cases_path, db_path)
        report.update(case_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def evaluate_cases(model_dir: Path, eval_cases_path: Path, db_path: Path | None) -> dict[str, Any]:
    cases = _load_jsonl(eval_cases_path)
    if not cases:
        return {"end_to_end_case_pass_rate": 0.0, "eval_cases": []}
    if db_path is None:
        return {"end_to_end_case_pass_rate": 0.0, "eval_cases": [{"id": row.get("id"), "passed": False, "reason": "db not provided"} for row in cases]}
    schema = read_sqlite_schema(db_path)
    predictor = OptionAIRPredictor(str(model_dir))
    rows = []
    passed = 0
    executed = 0
    for row in cases:
        result = predictor.predict(row["question"], schema)
        sql = result.get("sql") or ""
        sql_upper = sql.upper()
        contains_ok = all(str(fragment).upper() in sql_upper for fragment in row.get("expected_sql_contains", []))
        not_contains_ok = all(str(fragment).upper() not in sql_upper for fragment in row.get("expected_sql_not_contains", []))
        valid = bool((result.get("sql_validation") or {}).get("is_valid"))
        execution_ok = False
        if valid and sql:
            try:
                execute_select(db_path, sql, validation_result=result.get("sql_validation"))
                execution_ok = True
                executed += 1
            except Exception:
                execution_ok = False
        case_passed = valid and contains_ok and not_contains_ok
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
                "passed": case_passed,
                "sql": sql,
            }
        )
    return {
        "end_to_end_case_pass_rate": passed / max(len(cases), 1),
        "execution_success_rate": executed / max(len(cases), 1),
        "eval_cases": rows,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Option A QueryIR model.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "ir_test_examples.jsonl")
    parser.add_argument("--eval-cases", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR / "evaluation_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_option_a_model(args.model_dir, args.test, args.output, eval_cases_path=args.eval_cases, db_path=args.db)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
