from __future__ import annotations

from typing import Any


AGGREGATE_FUNCTIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


class SQLPatternClassifier:
    def classify(self, sql: str, features: dict[str, Any]) -> dict[str, Any]:
        if features.get("parse_error"):
            return self._unsupported("parse_error")
        if features.get("statement_type") != "SELECT":
            return self._unsupported("non_select")
        if features.get("has_nested_query"):
            return self._unsupported("nested_query")
        if features.get("has_set_operation"):
            return self._unsupported("set_operation")
        if features.get("has_window_function"):
            return self._unsupported("window_function")
        if features.get("has_having"):
            return self._unsupported("complex_having")

        aggregations = {str(item).upper() for item in features.get("aggregations", [])}
        group_by = features.get("group_by", [])
        order_by = features.get("order_by", [])
        has_limit = features.get("limit") is not None
        has_aggregation = bool(aggregations & AGGREGATE_FUNCTIONS)

        if self._is_trend(group_by):
            return self._supported("trend_by_date", "trend", 0.84)

        if has_aggregation and group_by and order_by and has_limit:
            desc = bool(order_by[0].get("desc"))
            if desc:
                return self._supported("top_n_metric_by_dimension", "rank_top", 0.92)
            return self._supported("bottom_n_metric_by_dimension", "rank_bottom", 0.92)

        if aggregations == {"COUNT"} and not group_by:
            return self._supported("count_records", "count", 0.9)
        if "COUNT" in aggregations and group_by:
            return self._supported("count_by_dimension", "count_by_dimension", 0.9)
        if aggregations & {"SUM", "AVG", "MIN", "MAX"} and not group_by:
            return self._supported("metric_summary", "metric_summary", 0.88)
        if aggregations & {"SUM", "AVG", "MIN", "MAX"} and group_by:
            return self._supported("metric_by_dimension", "metric_by_dimension", 0.88)
        if features.get("where_conditions") and not has_aggregation:
            return self._supported("simple_filter", "filter_records", 0.82)
        return self._supported("show_records", "show_records", 0.76)

    @staticmethod
    def _is_trend(group_by: list[str]) -> bool:
        date_markers = ["date", "time", "year", "month", "day", "strftime", "extract"]
        return any(any(marker in item.lower() for marker in date_markers) for item in group_by)

    @staticmethod
    def _supported(template_id: str, intent: str, confidence: float) -> dict[str, Any]:
        return {
            "is_supported": True,
            "template_id": template_id,
            "intent": intent,
            "unsupported_reason": None,
            "confidence": confidence,
        }

    @staticmethod
    def _unsupported(reason: str) -> dict[str, Any]:
        return {
            "is_supported": False,
            "template_id": None,
            "intent": None,
            "unsupported_reason": reason,
            "confidence": 0.0,
        }
