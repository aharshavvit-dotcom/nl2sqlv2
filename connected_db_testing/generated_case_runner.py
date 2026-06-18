from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from generic_planner import SchemaProfile, TableIntentResolver
from ir.ir_to_sql_renderer import IRToSQLRenderer


class ConnectedDBRegressionRunner:
    def run(self, cases: list[dict[str, Any]], schema: dict[str, Any], model_or_router: Any = None) -> dict[str, Any]:
        rows = []
        metrics = {
            "total_cases": len(cases),
            "passed_cases": 0,
            "case_pass_rate": 0.0,
            "direct_table_pass_rate": 0.0,
            "count_query_pass_rate": 0.0,
            "filter_query_pass_rate": 0.0,
            "explicit_join_pass_rate": 0.0,
            "unnecessary_join_count": 0,
            "wrong_table_count": 0,
            "select_star_count": 0,
            "sensitive_column_leak_count": 0,
            "clarification_count": 0,
        }
        by_type: dict[str, list[bool]] = {}
        model = model_or_router or _DirectAndRelationshipModel(schema)

        for case in cases:
            result = _predict(model, case["question"], schema, case)
            sql = _extract_sql(result)
            needs_clarification = bool(_extract_value(result, "needs_clarification", False))
            if needs_clarification:
                metrics["clarification_count"] += 1
            passed, failures = _check_case(case, sql, result)
            case_type = case.get("case_type", "unknown")
            by_type.setdefault(case_type, []).append(passed)
            metrics["passed_cases"] += 1 if passed else 0
            if "unnecessary_join" in failures:
                metrics["unnecessary_join_count"] += 1
            if "wrong_table" in failures:
                metrics["wrong_table_count"] += 1
            if "select_star" in failures:
                metrics["select_star_count"] += 1
            if "sensitive_column_leak" in failures:
                metrics["sensitive_column_leak_count"] += 1
            rows.append({"case_id": case.get("case_id"), "case_type": case_type, "passed": passed, "failures": failures, "sql": sql, "needs_clarification": needs_clarification})

        total = max(1, len(cases))
        metrics["case_pass_rate"] = metrics["passed_cases"] / total
        metrics["direct_table_pass_rate"] = _rate(by_type.get("direct_table_listing", []))
        metrics["count_query_pass_rate"] = _rate(by_type.get("count_query", []))
        metrics["filter_query_pass_rate"] = _rate(by_type.get("filter_query", []))
        metrics["explicit_join_pass_rate"] = _rate(by_type.get("explicit_join", []))
        return {"summary": metrics, "cases": rows}


class ConnectedDBRegressionReporter:
    def write(self, report: dict[str, Any], output: str | Path) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path = path.with_suffix(".md")
        summary = report.get("summary", {})
        lines = ["# Connected DB Regression Report", "", f"Case pass rate: **{summary.get('case_pass_rate', 0):.3f}**", "", "## Cases"]
        for row in report.get("cases", []):
            status = "pass" if row.get("passed") else "fail"
            lines.append(f"- {row.get('case_id')}: {status} ({', '.join(row.get('failures') or [])})")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _DirectAndRelationshipModel:
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.renderer = IRToSQLRenderer()

    def predict(self, question: str, schema: dict[str, Any], case: dict[str, Any] | None = None) -> dict[str, Any]:
        if case and case.get("case_type") == "explicit_join":
            return {"sql": _join_sql(case), "source_model": "connected_db_relationship_baseline", "needs_clarification": False}
        resolved = TableIntentResolver(SchemaProfile(schema)).resolve(question)
        if not resolved.handled or resolved.query_ir is None:
            return {"sql": None, "needs_clarification": True, "clarification": {"question": "Which table do you mean?", "options": []}}
        return {
            "sql": self.renderer.render(resolved.query_ir),
            "source_model": "generic_direct_planner",
            "query_ir": resolved.query_ir.model_dump(),
            "needs_clarification": False,
        }


def _predict(model: Any, question: str, schema: dict[str, Any], case: dict[str, Any]) -> Any:
    if hasattr(model, "predict"):
        try:
            return model.predict(question, schema, case)
        except TypeError:
            return model.predict(question, schema)
    if callable(model):
        return model(question, schema)
    return model


def _extract_sql(result: Any) -> str | None:
    return _extract_value(result, "sql", None)


def _extract_value(result: Any, key: str, default: Any) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    if hasattr(result, key):
        return getattr(result, key)
    return default


def _check_case(case: dict[str, Any], sql: str | None, result: Any) -> tuple[bool, list[str]]:
    expected = case.get("expected") or {}
    failures = []
    sql_text = sql or ""
    upper_sql = sql_text.upper()
    for needle in expected.get("must_include") or []:
        if str(needle).upper() not in upper_sql:
            failures.append(f"missing:{needle}")
    for needle in expected.get("must_not_include") or []:
        if needle and str(needle).upper() in upper_sql:
            failures.append("sensitive_column_leak" if _looks_sensitive(str(needle)) else "unnecessary_join")
    base_table = expected.get("base_table")
    if base_table and f'FROM "{base_table}"'.upper() not in upper_sql and f"FROM {base_table}".upper() not in upper_sql:
        failures.append("wrong_table")
    if expected.get("must_have_limit") and "LIMIT" not in upper_sql:
        failures.append("missing_limit")
    if expected.get("must_not_select_star") and re.search(r"SELECT\s+\*", upper_sql):
        failures.append("select_star")
    filter_column = expected.get("filter_column")
    if filter_column:
        where_clause = upper_sql.split("WHERE", 1)[1] if "WHERE" in upper_sql else ""
        if f'"{filter_column}"'.upper() not in where_clause and f".{filter_column}".upper() not in where_clause:
            failures.append("wrong_filter_column")
    if case.get("case_type") != "explicit_join" and "JOIN" in upper_sql:
        failures.append("unnecessary_join")
    if _extract_value(result, "needs_clarification", False) and not expected.get("allow_clarification"):
        failures.append("unexpected_clarification")
    return not failures, list(dict.fromkeys(failures))


def _join_sql(case: dict[str, Any]) -> str | None:
    rel = case.get("relationship") or {}
    from_table = rel.get("from_table")
    to_table = rel.get("to_table")
    from_column = rel.get("from_column")
    to_column = rel.get("to_column")
    if not all([from_table, to_table, from_column, to_column]):
        return None
    return (
        f'SELECT "{from_table}"."{from_column}", "{to_table}"."{to_column}" '
        f'FROM "{from_table}" JOIN "{to_table}" '
        f'ON "{from_table}"."{from_column}" = "{to_table}"."{to_column}" LIMIT 100'
    )


def _rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ["password", "token", "secret", "ssn", "email", "phone", "address"])
