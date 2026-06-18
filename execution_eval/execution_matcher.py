from __future__ import annotations

from typing import Any

from validation.sql_validator import SQLValidator

from .result_comparator import ResultComparator
from .sql_structure_comparator import SQLStructureComparator


class ExecutionMatcher:
    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit
        self.validator = SQLValidator()
        self.structure = SQLStructureComparator()
        self.results = ResultComparator()

    def evaluate_example(
        self,
        predicted_sql: str,
        gold_sql: str,
        schema: dict[str, Any],
        connector: Any,
        dialect: str,
    ) -> dict[str, Any]:
        predicted_validation = self.validator.validate(predicted_sql, schema=schema, max_limit=self.max_limit, dialect=dialect)
        gold_validation = self.validator.validate(gold_sql, schema=schema, max_limit=self.max_limit, dialect=dialect)
        structure = self.structure.compare(predicted_sql, gold_sql, schema=schema, dialect=dialect)
        payload = {
            "predicted_validation": predicted_validation,
            "gold_validation": gold_validation,
            "structure": structure,
            "execution_available": connector is not None,
            "executed": False,
            "execution_match": False,
            "result_comparison": None,
            "correct": False,
        }
        if not predicted_validation.get("is_valid") or not gold_validation.get("is_valid"):
            payload["skipped_reason"] = "sql_validation_failed"
            return payload
        if connector is None:
            payload["skipped_reason"] = "connector_unavailable"
            return payload

        pred_result = self._execute(connector, predicted_sql)
        gold_result = self._execute(connector, gold_sql)
        result_comparison = self.results.compare_results(pred_result, gold_result)
        payload.update(
            {
                "executed": True,
                "predicted_execution": pred_result,
                "gold_execution": gold_result,
                "result_comparison": result_comparison,
                "execution_match": result_comparison["result_match"],
                "correct": result_comparison["result_match"] and structure["structure_score"] >= 0.99,
            }
        )
        return payload

    @staticmethod
    def _execute(connector: Any, sql: str) -> dict[str, Any]:
        try:
            if hasattr(connector, "execute_readonly"):
                result = connector.execute_readonly(sql)
            elif hasattr(connector, "execute_query"):
                result = connector.execute_query(sql)
            elif hasattr(connector, "execute"):
                result = connector.execute(sql)
            elif callable(connector):
                result = connector(sql)
            else:
                raise TypeError("connector does not expose execute_readonly, execute_query, execute, or __call__")
            return _result_payload(result)
        except Exception as exc:
            return {"success": False, "columns": [], "rows": [], "error": str(exc)}


def _result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        if "success" not in result and "ok" not in result:
            result = {**result, "success": True}
        return result
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        return {
            "success": True,
            "columns": [str(column) for column in result.columns],
            "rows": result.to_dict(orient="records"),
        }
    if isinstance(result, list):
        columns = list(result[0].keys()) if result and isinstance(result[0], dict) else []
        return {"success": True, "columns": columns, "rows": result}
    return {"success": True, "columns": [], "rows": []}
