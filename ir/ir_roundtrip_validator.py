from __future__ import annotations

from typing import Any

import sqlglot

from validation.sql_validator import SQLValidator

from .ir_to_sql_renderer import IRToSQLRenderer
from .ir_validator import IRValidator
from .query_ir_models import QueryIR
from .sql_to_ir_rules import (
    date_grain_from_sql,
    detect_intent_from_ast,
    extract_aggregations,
    extract_group_by,
    extract_limit,
    extract_order_by,
    extract_tables,
    extract_where_filters,
)


class IRRoundtripValidator:
    def __init__(self, dialect: str = "sqlite", max_limit: int = 1000):
        self.dialect = dialect
        self.max_limit = max_limit
        self.ir_validator = IRValidator(max_limit=max_limit)
        self.sql_validator = SQLValidator()

    def validate_roundtrip(
        self,
        source_sql: str,
        query_ir: QueryIR,
        rendered_sql: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        issues: list[str] = []
        ir_validation = self.ir_validator.validate(query_ir, schema=schema)
        sql_validation = self.sql_validator.validate(rendered_sql, schema=schema, max_limit=self.max_limit, dialect=self.dialect)

        source_features = self._features(source_sql)
        rendered_features = self._features(rendered_sql)

        checks = {
            "ir_valid": bool(ir_validation.is_valid),
            "rendered_sql_valid": bool(sql_validation.get("is_valid")),
            "intent_compatible": self._intent_compatible(query_ir, source_features),
            "tables_compatible": self._tables_compatible(query_ir, source_features, rendered_features),
            "metrics_compatible": self._metrics_compatible(query_ir, source_features),
            "dimensions_compatible": self._dimensions_compatible(query_ir, source_features),
            "filters_compatible": self._filters_compatible(query_ir, source_features),
            "group_order_limit_compatible": self._group_order_limit_compatible(query_ir, source_features),
        }
        for name, passed in checks.items():
            if not passed:
                issues.append(f"{name} failed")
        issues.extend(ir_validation.errors)
        issues.extend(str(issue) for issue in sql_validation.get("issues", []))
        return {
            "is_valid": all(checks.values()),
            "checks": checks,
            "issues": list(dict.fromkeys(issues)),
        }

    def _features(self, sql: str) -> dict[str, Any]:
        try:
            ast = sqlglot.parse_one(sql, read=self.dialect)
        except Exception:
            return {"parse_error": True}
        aggregations = extract_aggregations(ast)
        group_by = extract_group_by(ast)
        order_by = extract_order_by(ast)
        where_filters = extract_where_filters(ast)
        return {
            "parse_error": False,
            "ast": ast,
            "tables": extract_tables(ast),
            "aggregations": aggregations,
            "group_by": group_by,
            "order_by": order_by,
            "where_filters": where_filters,
            "limit": extract_limit(ast),
            "intent": detect_intent_from_ast(
                ast,
                {
                    "aggregations": aggregations,
                    "group_by": group_by,
                    "order_by": order_by,
                    "where_filters": where_filters,
                    "limit": extract_limit(ast),
                },
            ),
        }

    @staticmethod
    def _intent_compatible(query_ir: QueryIR, source_features: dict[str, Any]) -> bool:
        if source_features.get("parse_error"):
            return False
        return source_features.get("intent") == query_ir.intent

    @staticmethod
    def _tables_compatible(query_ir: QueryIR, source_features: dict[str, Any], rendered_features: dict[str, Any]) -> bool:
        if source_features.get("parse_error") or rendered_features.get("parse_error"):
            return False
        source_tables = set(source_features.get("tables") or [])
        rendered_tables = set(rendered_features.get("tables") or [])
        required_tables = set(query_ir.required_tables or [])
        return source_tables == rendered_tables and source_tables.issubset(required_tables | {query_ir.base_table})

    @staticmethod
    def _metrics_compatible(query_ir: QueryIR, source_features: dict[str, Any]) -> bool:
        source_aggs = source_features.get("aggregations") or []
        if not query_ir.metrics and not source_aggs:
            return True
        if len(query_ir.metrics) != len(source_aggs):
            return False
        for metric, source in zip(query_ir.metrics, source_aggs):
            if metric.aggregation.upper() != str(source.get("function", "")).upper():
                return False
            if metric.expression != "*" and normalize_sql_fragment(metric.expression) not in normalize_sql_fragment(str(source.get("argument_sql"))):
                return False
        return True

    @staticmethod
    def _dimensions_compatible(query_ir: QueryIR, source_features: dict[str, Any]) -> bool:
        group_by = [normalize_group_expression(item) for item in source_features.get("group_by") or []]
        ir_group_by = [normalize_group_expression(item) for item in query_ir.group_by]
        if query_ir.intent == "trend_by_date":
            return bool(group_by) and bool(ir_group_by) and group_by == ir_group_by
        if query_ir.intent in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"}:
            return bool(group_by) and group_by == ir_group_by
        return True

    @staticmethod
    def _filters_compatible(query_ir: QueryIR, source_features: dict[str, Any]) -> bool:
        source_filters = source_features.get("where_filters") or []
        ir_filter_count = len(query_ir.filters)
        date_range_filters = [item for item in query_ir.date_filters if item.filter_type != "grain"]
        if not source_filters:
            return ir_filter_count == 0 and not date_range_filters
        source_non_date = [item for item in source_filters if not is_date_filter(item)]
        source_date = [item for item in source_filters if is_date_filter(item)]
        return len(source_non_date) == ir_filter_count and (not source_date or bool(date_range_filters))

    @staticmethod
    def _group_order_limit_compatible(query_ir: QueryIR, source_features: dict[str, Any]) -> bool:
        source_limit = source_features.get("limit")
        limit_ok = source_limit is None or source_limit == query_ir.limit
        source_order = source_features.get("order_by") or []
        order_ok = True if not source_order else bool(query_ir.order_by)
        return limit_ok and order_ok


def normalize_sql_fragment(value: str) -> str:
    return " ".join(str(value).lower().split())


def normalize_group_expression(value: str) -> str:
    grain = date_grain_from_sql(value)
    if grain:
        return f"date_grain({grain['date_expression']},{grain['date_grain']})"
    if value.startswith("DATE_GRAIN("):
        return value.lower().replace(" ", "")
    return normalize_sql_fragment(value)


def is_date_filter(item: dict[str, Any]) -> bool:
    left = item.get("left") or {}
    column = str(left.get("column") or "").lower()
    value = item.get("value")
    return "date" in column or (isinstance(value, str) and len(value) >= 10 and value[4:5] == "-" and value[7:8] == "-")

