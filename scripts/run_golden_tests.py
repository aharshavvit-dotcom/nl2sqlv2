from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.query_executor import execute_select
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel


ARTIFACT_DIR = ROOT / "artifacts" / "option_c_model"
DB_PATH = ROOT / "data" / "sample_retail.db"
GOLDEN_PATH = ROOT / "evaluation" / "golden_tests.jsonl"
RESULTS_PATH = ROOT / "evaluation" / "golden_test_results.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run() -> dict[str, Any]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Sample database not found: {DB_PATH}. Run python scripts/create_sample_db.py first.")
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden test file not found: {GOLDEN_PATH}")

    schema = read_sqlite_schema(DB_PATH)
    model = RetrievalNL2SQLModel.load(artifact_dir=ARTIFACT_DIR)
    tests = load_jsonl(GOLDEN_PATH)
    results = []

    for item in tests:
        result = model.predict(item["question"], schema)
        sql = result.sql or ""
        sql_lower = sql.lower()
        expected_template = item.get("expected_template_id")
        if expected_template is None:
            template_ok = result.confidence < 0.50 and result.confidence_tier == "low"
        else:
            template_ok = result.template_id == expected_template
        contains_ok = all(str(fragment).lower() in sql_lower for fragment in item.get("expected_sql_contains", []))
        excludes_ok = all(str(fragment).lower() not in sql_lower for fragment in item.get("expected_sql_excludes", []))
        execution_ok = True
        execution_error = None
        if result.validation.get("is_valid", result.validation.get("ok")) and sql:
            try:
                execute_select(DB_PATH, sql, validation_result=result.validation)
            except Exception as exc:  # pragma: no cover - reported in result payload
                execution_ok = False
                execution_error = str(exc)
        passed = bool(template_ok and contains_ok and excludes_ok and execution_ok)
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "passed": passed,
                "expected_template_id": expected_template,
                "actual_template_id": result.template_id,
                "confidence": result.confidence,
                "confidence_tier": result.confidence_tier,
                "validation_ok": result.validation.get("is_valid", result.validation.get("ok")),
                "template_ok": template_ok,
                "contains_ok": contains_ok,
                "excludes_ok": excludes_ok,
                "execution_ok": execution_ok,
                "execution_error": execution_error,
                "sql": sql,
            }
        )

    passed_count = sum(1 for row in results if row["passed"])
    payload = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "accuracy": passed_count / len(results) if results else 0.0,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    payload = run()
    print(f"Golden tests: {payload['passed']}/{payload['total']} passed ({payload['accuracy']:.1%})")
    for row in payload["results"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['id']} {row['question']}")
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
