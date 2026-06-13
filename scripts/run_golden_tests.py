from __future__ import annotations

import argparse
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


DEFAULT_DB = ROOT / "data" / "sample_retail.db"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "option_c_model"
DEFAULT_GOLDEN_FILE = ROOT / "evaluation" / "golden_runtime_tests.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation" / "golden_runtime_report.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_golden_tests(
    db_path: Path = DEFAULT_DB,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    golden_file: Path = DEFAULT_GOLDEN_FILE,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"Sample database not found: {db_path}. Run python scripts/create_sample_db.py first.")
    if not golden_file.exists():
        raise FileNotFoundError(f"Golden test file not found: {golden_file}")

    schema = read_sqlite_schema(db_path)
    model = RetrievalNL2SQLModel.load(artifact_dir=artifact_dir)
    cases = load_jsonl(golden_file)
    case_results = []

    for case in cases:
        result = model.predict(case["question"], schema)
        case_result = evaluate_case(case, result, db_path)
        case_results.append(case_result)

    passed_cases = sum(1 for row in case_results if row["passed"])
    total_cases = len(case_results)
    payload = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "accuracy": passed_cases / total_cases if total_cases else 0.0,
        "sql_validity_rate": _rate(case_results, "sql_validation"),
        "execution_success_rate": _rate(case_results, "execution"),
        "query_ir_match_rate": _query_ir_rate(case_results),
        "failures_by_category": _failure_counts(case_results),
        "case_results": case_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def evaluate_case(case: dict[str, Any], result: Any, db_path: Path) -> dict[str, Any]:
    sql = result.sql or ""
    query_ir = result.query_ir or {}
    expected_ir = case.get("expected_query_ir") or {}
    checks: dict[str, dict[str, Any]] = {}

    _check(checks, "sql_exists", bool(sql), "sql_validation", "SQL was not generated")
    _check(checks, "query_ir_exists", bool(query_ir), "missing_query_ir", "QueryIR was not generated")
    _check(checks, "ir_validation", bool((result.ir_validation or {}).get("is_valid")), "ir_validation_failed", "IR validation failed")
    _check(checks, "sql_validation", bool((result.validation or {}).get("is_valid")), "sql_validation_failed", "SQL validation failed")

    if case.get("expected_template_id"):
        _check(
            checks,
            "template",
            result.template_id == case["expected_template_id"],
            "wrong_template",
            f"Expected template {case['expected_template_id']}, got {result.template_id}",
        )
    if case.get("expected_intent"):
        _check(
            checks,
            "intent",
            query_ir.get("intent") == case["expected_intent"],
            "wrong_template",
            f"Expected intent {case['expected_intent']}, got {query_ir.get('intent')}",
        )

    _check_expected_ir(checks, query_ir, expected_ir)
    _check_sql_fragments(checks, sql, case)
    execution_payload = _execute_if_requested(case, result, db_path)
    _check(
        checks,
        "execution",
        execution_payload["ok"],
        "execution_failed",
        execution_payload.get("error") or "Execution failed",
        skipped=execution_payload["skipped"],
    )

    if case.get("expected_result_columns") and not execution_payload["skipped"] and execution_payload.get("columns") is not None:
        missing = [column for column in case["expected_result_columns"] if column not in execution_payload["columns"]]
        _check(
            checks,
            "result_columns",
            not missing,
            "execution_failed",
            "Missing result columns: " + ", ".join(missing),
        )

    passed = all(check["passed"] or check.get("skipped") for check in checks.values())
    return {
        "id": case["id"],
        "question": case["question"],
        "passed": passed,
        "checks": checks,
        "failure_categories": sorted({check["category"] for check in checks.values() if not check["passed"] and not check.get("skipped")}),
        "actual_template_id": result.template_id,
        "confidence": result.confidence,
        "confidence_tier": result.confidence_tier,
        "sql": sql,
        "query_ir": query_ir,
    }


def _check_expected_ir(checks: dict[str, dict[str, Any]], query_ir: dict[str, Any], expected_ir: dict[str, Any]) -> None:
    if not expected_ir:
        return
    if "base_table" in expected_ir:
        _check(checks, "base_table", query_ir.get("base_table") == expected_ir["base_table"], "wrong_template", "Base table mismatch")
    if "required_tables" in expected_ir:
        actual = set(query_ir.get("required_tables") or [])
        expected = set(expected_ir["required_tables"])
        _check(checks, "required_tables", expected.issubset(actual), "missing_expected_join", "Required tables missing")
    if "metric_alias" in expected_ir:
        metric = _first(query_ir.get("metrics"))
        _check(checks, "metric_alias", metric.get("alias") == expected_ir["metric_alias"], "missing_expected_metric", "Metric alias mismatch")
    if "metric_aggregation" in expected_ir:
        metric = _first(query_ir.get("metrics"))
        _check(
            checks,
            "metric_aggregation",
            metric.get("aggregation") == expected_ir["metric_aggregation"],
            "missing_expected_metric",
            "Metric aggregation mismatch",
        )
    if "metric_expression_contains" in expected_ir:
        metric = _first(query_ir.get("metrics"))
        fragment = expected_ir["metric_expression_contains"]
        _check(
            checks,
            "metric_expression",
            fragment in (metric.get("expression") or ""),
            "missing_expected_metric",
            f"Metric expression missing {fragment}",
        )
    if "dimension_expression" in expected_ir:
        dimension = _first(query_ir.get("dimensions"))
        _check(
            checks,
            "dimension_expression",
            dimension.get("expression") == expected_ir["dimension_expression"],
            "missing_expected_dimension",
            "Dimension expression mismatch",
        )
    if "join_conditions" in expected_ir:
        actual = {join.get("condition") for join in query_ir.get("joins", [])}
        expected = set(expected_ir["join_conditions"])
        _check(checks, "join_conditions", expected.issubset(actual), "missing_expected_join", "Join condition missing")
    if expected_ir.get("filter_expression"):
        actual = {item.get("expression") for item in query_ir.get("filters", [])}
        _check(
            checks,
            "filter_expression",
            expected_ir["filter_expression"] in actual,
            "missing_expected_filter",
            "Expected filter missing",
        )
    if expected_ir.get("date_filter_exists"):
        _check(
            checks,
            "date_filter",
            bool(query_ir.get("date_filters")),
            "missing_expected_date_filter",
            "Expected date filter missing",
        )
    if "limit" in expected_ir:
        _check(checks, "limit", query_ir.get("limit") == expected_ir["limit"], "wrong_template", "Limit mismatch")


def _check_sql_fragments(checks: dict[str, dict[str, Any]], sql: str, case: dict[str, Any]) -> None:
    for index, fragment in enumerate(case.get("expected_sql_contains", [])):
        _check(
            checks,
            f"sql_contains_{index}",
            str(fragment) in sql,
            "sql_validation_failed",
            f"SQL missing fragment: {fragment}",
        )
    forbidden = case.get("expected_sql_not_contains") or case.get("expected_sql_excludes") or []
    for index, fragment in enumerate(forbidden):
        _check(
            checks,
            f"sql_not_contains_{index}",
            str(fragment) not in sql,
            "semantic_grain_risk" if "orders.amount" in str(fragment) else "sql_validation_failed",
            f"SQL contained forbidden fragment: {fragment}",
        )


def _execute_if_requested(case: dict[str, Any], result: Any, db_path: Path) -> dict[str, Any]:
    if not case.get("should_execute", False):
        return {"ok": True, "skipped": True, "columns": None}
    if not result.sql or not (result.validation or {}).get("is_valid"):
        return {"ok": False, "skipped": False, "error": "No valid SQL to execute", "columns": None}
    try:
        df = execute_select(db_path, result.sql, validation_result=result.validation)
    except Exception as exc:
        return {"ok": False, "skipped": False, "error": str(exc), "columns": None}
    return {"ok": True, "skipped": False, "columns": list(df.columns)}


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    category: str,
    message: str,
    skipped: bool = False,
) -> None:
    checks[name] = {"passed": bool(passed), "category": category, "message": "" if passed else message, "skipped": skipped}


def _first(items: Any) -> dict[str, Any]:
    return items[0] if isinstance(items, list) and items else {}


def _rate(case_results: list[dict[str, Any]], check_name: str) -> float:
    if not case_results:
        return 0.0
    return sum(1 for row in case_results if row["checks"].get(check_name, {}).get("passed")) / len(case_results)


def _query_ir_rate(case_results: list[dict[str, Any]]) -> float:
    if not case_results:
        return 0.0
    ir_checks = {
        "query_ir_exists",
        "ir_validation",
        "template",
        "intent",
        "base_table",
        "required_tables",
        "metric_alias",
        "metric_aggregation",
        "metric_expression",
        "dimension_expression",
        "join_conditions",
        "filter_expression",
        "date_filter",
        "limit",
    }
    matched = 0
    for row in case_results:
        relevant = [check for name, check in row["checks"].items() if name in ir_checks]
        if relevant and all(check["passed"] for check in relevant):
            matched += 1
    return matched / len(case_results)


def _failure_counts(case_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in case_results:
        for category in row["failure_categories"]:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QueryIR runtime golden tests.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--golden-file", type=Path, default=DEFAULT_GOLDEN_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_golden_tests(args.db, args.artifact_dir, args.golden_file, args.output)
    print(f"Golden tests: {payload['passed_cases']}/{payload['total_cases']} passed ({payload['accuracy']:.1%})")
    for row in payload["case_results"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['id']} {row['question']}")
    return 0 if payload["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
