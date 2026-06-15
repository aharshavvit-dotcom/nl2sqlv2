from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from execution.query_executor import execute_select


PredictFn = Callable[[str], dict[str, Any]]


@dataclass
class HybridBenchmark:
    option_c_predictor: PredictFn | None = None
    option_a_predictor: PredictFn | None = None
    hybrid_predictor: PredictFn | None = None

    def run(self, cases: list[dict], db_path: str | None = None) -> dict:
        option_c_results = self._run_model("option_c", self.option_c_predictor, cases, db_path)
        option_a_results = self._run_model("option_a", self.option_a_predictor, cases, db_path)
        hybrid_results = self._run_model("hybrid", self.hybrid_predictor, cases, db_path)
        option_c_summary = _summarize(option_c_results)
        option_a_summary = _summarize(option_a_results)
        hybrid_summary = _summarize(hybrid_results)
        return {
            "option_c": option_c_summary,
            "option_a": option_a_summary,
            "hybrid": hybrid_summary,
            "comparison": {
                "hybrid_gain_over_option_c": hybrid_summary["case_pass_rate"] - option_c_summary["case_pass_rate"],
                "hybrid_gain_over_option_a": hybrid_summary["case_pass_rate"] - option_a_summary["case_pass_rate"],
            },
            "case_results": [
                {
                    "id": case.get("id"),
                    "question": case.get("question"),
                    "option_c": option_c_results[idx],
                    "option_a": option_a_results[idx],
                    "hybrid": hybrid_results[idx],
                }
                for idx, case in enumerate(cases)
            ],
        }

    def _run_model(self, name: str, predictor: PredictFn | None, cases: list[dict], db_path: str | None) -> list[dict[str, Any]]:
        rows = []
        for case in cases:
            if predictor is None:
                rows.append(_missing_result(name, case))
                continue
            try:
                result = predictor(case["question"])
            except Exception as exc:
                rows.append({"model": name, "id": case.get("id"), "passed": False, "error": str(exc), "sql_valid": False, "execution_success": False, "confidence": 0.0})
                continue
            rows.append(_evaluate_case(name, case, result, db_path))
        return rows


def _evaluate_case(name: str, case: dict[str, Any], result: dict[str, Any], db_path: str | None) -> dict[str, Any]:
    validation = result.get("validation") or result.get("sql_validation") or {}
    sql = result.get("sql") or ""
    sql_upper = sql.upper()
    sql_valid = bool(validation.get("is_valid", validation.get("ok", False)))
    contains_ok = all(str(fragment).upper() in sql_upper for fragment in case.get("expected_sql_contains", []))
    not_contains_ok = all(str(fragment).upper() not in sql_upper for fragment in case.get("expected_sql_not_contains", []))
    execution_success = False
    if db_path and sql_valid and sql and case.get("should_execute", True):
        try:
            execute_select(db_path, sql, validation_result=validation)
            execution_success = True
        except Exception:
            execution_success = False
    passed = sql_valid and contains_ok and not_contains_ok and (execution_success or not db_path or not case.get("should_execute", True))
    return {
        "model": name,
        "id": case.get("id"),
        "passed": passed,
        "sql_valid": sql_valid,
        "execution_success": execution_success,
        "confidence": float(result.get("confidence") or 0.0),
        "sql": sql,
        "router_decision": result.get("router_decision") or (result.get("debug") or {}).get("router_decision", {}),
        "error_counts": _error_counts(case, sql, sql_valid),
    }


def _missing_result(name: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        "id": case.get("id"),
        "passed": False,
        "sql_valid": False,
        "execution_success": False,
        "confidence": 0.0,
        "sql": "",
        "router_decision": {},
        "error_counts": {"invalid_sql_count": 1},
        "missing": True,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(rows), 1)
    error_totals = {
        "invalid_sql_count": 0,
        "wrong_metric_count": 0,
        "wrong_dimension_count": 0,
        "wrong_filter_count": 0,
        "wrong_date_filter_count": 0,
        "wrong_join_count": 0,
    }
    for row in rows:
        for key, value in (row.get("error_counts") or {}).items():
            error_totals[key] = error_totals.get(key, 0) + int(value)
    return {
        "case_pass_rate": sum(1 for row in rows if row.get("passed")) / total,
        "sql_validation_rate": sum(1 for row in rows if row.get("sql_valid")) / total,
        "execution_success_rate": sum(1 for row in rows if row.get("execution_success")) / total,
        "average_confidence": sum(float(row.get("confidence") or 0.0) for row in rows) / total,
        **error_totals,
    }


def _error_counts(case: dict[str, Any], sql: str, sql_valid: bool) -> dict[str, int]:
    upper_sql = sql.upper()
    counts = {
        "invalid_sql_count": 0 if sql_valid else 1,
        "wrong_metric_count": 0,
        "wrong_dimension_count": 0,
        "wrong_filter_count": 0,
        "wrong_date_filter_count": 0,
        "wrong_join_count": 0,
    }
    expected_metric = str(case.get("expected_metric_contains") or "").upper()
    expected_dimension = str(case.get("expected_dimension_contains") or "").upper()
    if expected_metric and expected_metric not in upper_sql:
        counts["wrong_metric_count"] = 1
    if expected_dimension and expected_dimension not in upper_sql:
        counts["wrong_dimension_count"] = 1
    if any(token in str(case.get("id", "")).lower() for token in ["filter", "status", "region", "category"]) and "WHERE" not in upper_sql:
        counts["wrong_filter_count"] = 1
    if any(token in str(case.get("question", "")).lower() for token in ["month", "year", "last 30", "last month"]) and ("DATE" not in upper_sql and "STRFTIME" not in upper_sql):
        counts["wrong_date_filter_count"] = 1
    if "JOIN" not in upper_sql and "." in upper_sql and expected_dimension and expected_metric:
        counts["wrong_join_count"] = 1
    return counts
