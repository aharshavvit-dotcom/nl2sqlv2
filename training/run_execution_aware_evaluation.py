from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
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


def evaluate_controlled_fixtures(
    fixture_sql_path: Path | None = None,
    fixture_cases_path: Path | None = None,
) -> dict[str, Any]:
    """Run controlled execution-aware evaluation using a known fixture DB.

    Creates a temporary SQLite database from the fixture SQL seed,
    executes gold SQL for each case, and verifies row counts and safety.
    """
    fixture_dir = ROOT / "evaluation" / "fixtures"
    sql_path = fixture_sql_path or fixture_dir / "controlled_evaluation.sql"
    cases_path = fixture_cases_path or fixture_dir / "controlled_evaluation_cases.jsonl"

    if not sql_path.exists():
        raise FileNotFoundError(f"Fixture SQL not found: {sql_path}")
    if not cases_path.exists():
        raise FileNotFoundError(f"Fixture cases not found: {cases_path}")

    sql_seed = sql_path.read_text(encoding="utf-8")
    cases = read_jsonl(cases_path)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "controlled_evaluation.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(sql_seed)
            for case in cases:
                gold_sql = case.get("gold_sql", "")
                example_id = case.get("example_id", "")
                expected_rows = case.get("expected_row_count")
                entry: dict[str, Any] = {
                    "example_id": example_id,
                    "question": case.get("question"),
                    "gold_sql": gold_sql,
                    "expected_row_count": expected_rows,
                    "execution_success": False,
                    "actual_row_count": None,
                    "row_count_match": False,
                    "sql_is_select_only": gold_sql.strip().upper().startswith("SELECT"),
                    "error": None,
                }
                try:
                    cursor = conn.execute(gold_sql)
                    rows = cursor.fetchall()
                    entry["execution_success"] = True
                    entry["actual_row_count"] = len(rows)
                    if expected_rows is not None:
                        entry["row_count_match"] = len(rows) == expected_rows
                except Exception as exc:
                    entry["error"] = str(exc)
                results.append(entry)
        finally:
            conn.close()

    total = len(results)
    exec_ok = sum(1 for r in results if r["execution_success"])
    row_match = sum(1 for r in results if r["row_count_match"])
    select_only = sum(1 for r in results if r["sql_is_select_only"])

    return {
        "controlled_fixture_evaluation": True,
        "evaluation_type": "controlled_gold_sql_fixture_validation",
        "measures_model_predictions": False,
        "fixture_sql": str(sql_path),
        "fixture_cases": str(cases_path),
        "total_cases": total,
        "summary": {
            "execution_success_rate": exec_ok / total if total else 0.0,
            "row_count_match_rate": row_match / total if total else 0.0,
            "select_only_rate": select_only / total if total else 0.0,
        },
        "cases": results,
    }


def evaluate_controlled_predicted_sql(
    model_artifact_dir: Path | None = None,
    fixture_sql_path: Path | None = None,
    fixture_cases_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run predicted-SQL controlled execution evaluation.

    Unlike evaluate_controlled_fixtures (which validates gold SQL),
    this function loads the trained model, generates predictions for each
    fixture question, and evaluates whether the predicted SQL executes
    correctly against the controlled fixture database.

    This is the stronger model metric: it measures actual model prediction accuracy.
    """
    fixture_dir = ROOT / "evaluation" / "fixtures"
    sql_path = fixture_sql_path or fixture_dir / "controlled_evaluation.sql"
    cases_path = fixture_cases_path or fixture_dir / "controlled_evaluation_cases.jsonl"

    if not sql_path.exists():
        return {"error": f"Fixture SQL not found: {sql_path}", "evaluation_type": "controlled_predicted_sql_execution"}
    if not cases_path.exists():
        return {"error": f"Fixture cases not found: {cases_path}", "evaluation_type": "controlled_predicted_sql_execution"}

    # Resolve model artifact directory
    artifact_dir = model_artifact_dir
    if artifact_dir is None:
        # Try candidate bundle, then current bundle, then raw artifacts
        for candidate_path in [
            ROOT / "artifacts" / "model_bundle" / "candidate",
            ROOT / "artifacts" / "model_bundle" / "current",
        ]:
            if (candidate_path / "retrieval_ir").exists():
                artifact_dir = candidate_path
                break
    if artifact_dir is None or not artifact_dir.exists():
        return {
            "error": "No model artifact directory found for predicted-SQL evaluation",
            "evaluation_type": "controlled_predicted_sql_execution",
            "measures_model_predictions": True,
        }

    sql_seed = sql_path.read_text(encoding="utf-8")
    cases = read_jsonl(cases_path)

    # Load model
    try:
        from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
        model = RetrievalNL2SQLModel.load(str(artifact_dir))
    except Exception as exc:
        return {
            "error": f"Failed to load model from {artifact_dir}: {exc}",
            "evaluation_type": "controlled_predicted_sql_execution",
            "measures_model_predictions": True,
        }

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "controlled_predicted.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(sql_seed)

            # Execute gold SQL first to get expected results
            gold_results: dict[str, list[tuple]] = {}
            for case in cases:
                gold_sql = case.get("gold_sql", "")
                example_id = case.get("example_id", "")
                try:
                    cursor = conn.execute(gold_sql)
                    gold_results[example_id] = cursor.fetchall()
                except Exception:
                    gold_results[example_id] = []

            for case in cases:
                example_id = case.get("example_id", "")
                question = case.get("question", "")
                gold_sql = case.get("gold_sql", "")
                expected_rows = case.get("expected_row_count")

                entry: dict[str, Any] = {
                    "example_id": example_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "expected_row_count": expected_rows,
                    "predicted_sql": None,
                    "prediction_generated": False,
                    "predicted_sql_valid": False,
                    "predicted_sql_is_select_only": False,
                    "predicted_execution_success": False,
                    "predicted_actual_row_count": None,
                    "predicted_row_count_match": False,
                    "predicted_result_value_match": False,
                    "error": None,
                }

                # Generate prediction
                try:
                    from schema_tools.schema_graph import SchemaGraph
                    # Build minimal schema from fixture
                    schema = SchemaGraph(tables={}, relationships=[])
                    result = model.predict(question, schema)
                    predicted_sql = result.sql or ""
                    entry["predicted_sql"] = predicted_sql
                    entry["prediction_generated"] = bool(predicted_sql.strip())

                    if predicted_sql.strip():
                        # Validate safety
                        sql_upper = predicted_sql.strip().upper()
                        entry["predicted_sql_is_select_only"] = sql_upper.startswith("SELECT")
                        entry["predicted_sql_valid"] = (
                            entry["predicted_sql_is_select_only"]
                            and not any(kw in sql_upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"])
                        )

                        if entry["predicted_sql_valid"]:
                            try:
                                cursor = conn.execute(predicted_sql)
                                pred_rows = cursor.fetchall()
                                entry["predicted_execution_success"] = True
                                entry["predicted_actual_row_count"] = len(pred_rows)
                                if expected_rows is not None:
                                    entry["predicted_row_count_match"] = len(pred_rows) == expected_rows
                                # Compare with gold results
                                gold_rows = gold_results.get(example_id, [])
                                entry["predicted_result_value_match"] = (pred_rows == gold_rows)
                            except Exception as exc:
                                entry["error"] = f"Execution error: {exc}"
                except Exception as exc:
                    entry["error"] = f"Prediction error: {exc}"

                results.append(entry)
        finally:
            conn.close()

    total = len(results)
    predictions_generated = sum(1 for r in results if r["prediction_generated"])
    valid_count = sum(1 for r in results if r["predicted_sql_valid"])
    exec_success = sum(1 for r in results if r["predicted_execution_success"])
    exec_match = sum(1 for r in results if r["predicted_result_value_match"])
    row_match = sum(1 for r in results if r["predicted_row_count_match"])
    unsafe = sum(1 for r in results if r["prediction_generated"] and not r["predicted_sql_is_select_only"])

    return {
        "evaluation_type": "controlled_predicted_sql_execution",
        "measures_model_predictions": True,
        "model_artifact_dir": str(artifact_dir),
        "fixture_sql": str(sql_path),
        "fixture_cases": str(cases_path),
        "cases_total": total,
        "predictions_generated": predictions_generated,
        "predicted_sql_valid_count": valid_count,
        "predicted_execution_success_count": exec_success,
        "predicted_execution_match_count": exec_match,
        "predicted_execution_match_rate": exec_match / total if total else 0.0,
        "predicted_execution_error_rate": (total - exec_success) / total if total else 0.0,
        "predicted_row_count_match_rate": row_match / total if total else 0.0,
        "predicted_result_value_match_rate": exec_match / total if total else 0.0,
        "unsafe_sql_count": unsafe,
        "passed": unsafe == 0 and (exec_match / total if total else 0.0) >= (config or {}).get("min_execution_match_rate", 0.0),
        "cases": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-aware evaluation on prediction rows.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "execution_aware_evaluation_report.json")
    parser.add_argument("--run-controlled-fixtures", action="store_true", help="Run controlled fixture evaluation instead of prediction rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_controlled_fixtures:
        report = evaluate_controlled_fixtures()
        write_json(args.output, report)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
        return 0
    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}. Run training/evaluate_against_gold.py first.")
    rows = read_jsonl(args.predictions)
    report = evaluate_rows(rows)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
