from __future__ import annotations

import re
from typing import Any

from .query_ir_models import IRDateFilter, IRFilter, IRMetric, QueryIR


SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address")


class IRToSQLRenderer:
    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit

    def render(self, query_ir: QueryIR) -> str:
        parts = [
            self.render_select(query_ir),
            self.render_from(query_ir),
            self.render_joins(query_ir),
            self.render_where(query_ir),
            self.render_group_by(query_ir),
            self.render_order_by(query_ir),
            self.render_limit(query_ir),
        ]
        return self.clean_sql("\n".join(part for part in parts if part))

    def render_select(self, query_ir: QueryIR) -> str:
        template_id = query_ir.template_id
        metric = query_ir.metrics[0] if query_ir.metrics else None
        dimension = query_ir.dimensions[0] if query_ir.dimensions else None

        if template_id == "count_records" or (query_ir.select_mode == "count" and not dimension):
            return "SELECT\n  COUNT(*) AS record_count"
        if template_id == "count_by_dimension" and dimension:
            return f"SELECT\n  {dimension.expression} AS {dimension.alias},\n  COUNT(*) AS record_count"
        if template_id == "trend_by_date" and metric:
            grain = self._grain_filter(query_ir) or self._first_date_filter(query_ir)
            if grain:
                date_expr = self.render_date_grain(grain.date_expression, grain.date_grain or "month")
                return f"SELECT\n  {date_expr} AS period,\n  {self._metric_sql(metric)} AS {metric.alias}"
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"} and metric and dimension:
            return f"SELECT\n  {dimension.expression} AS {dimension.alias},\n  {self._metric_sql(metric)} AS {metric.alias}"
        if template_id == "metric_summary" and metric:
            return f"SELECT\n  {self._metric_sql(metric)} AS {metric.alias}"

        record_columns = self._record_select_columns(query_ir)
        return "SELECT\n  " + ",\n  ".join(record_columns)

    @staticmethod
    def render_from(query_ir: QueryIR) -> str:
        return f"FROM {query_ir.base_table}" if query_ir.base_table else ""

    @staticmethod
    def render_joins(query_ir: QueryIR) -> str:
        lines = []
        for join in sorted(query_ir.joins, key=lambda item: item.path_order):
            join_type = (join.join_type or "INNER").upper()
            lines.append(f"{join_type} JOIN {join.right_table}\n  ON {join.condition}")
        return "\n".join(lines)

    def render_where(self, query_ir: QueryIR) -> str:
        clauses = [self._filter_sql(item) for item in query_ir.filters]
        clauses.extend(self._date_filter_sql(item) for item in query_ir.date_filters if item.filter_type != "grain")
        clauses = [clause for clause in clauses if clause]
        if not clauses:
            return ""
        return "WHERE " + "\n  AND ".join(clauses)

    def render_group_by(self, query_ir: QueryIR) -> str:
        expressions = [self._render_group_expression(query_ir, expression) for expression in query_ir.group_by]
        expressions = [expression for expression in expressions if expression]
        return "GROUP BY " + ", ".join(expressions) if expressions else ""

    @staticmethod
    def render_order_by(query_ir: QueryIR) -> str:
        if not query_ir.order_by:
            return ""
        parts = [f"{item.alias or item.expression} {item.direction}" for item in query_ir.order_by]
        return "ORDER BY " + ", ".join(parts)

    def render_limit(self, query_ir: QueryIR) -> str:
        return f"LIMIT {min(max(int(query_ir.limit or 100), 1), self.max_limit)}"

    @staticmethod
    def render_date_grain(date_expression: str, grain: str) -> str:
        return f"strftime('%Y', {date_expression})" if grain == "year" else f"strftime('%Y-%m', {date_expression})"

    @staticmethod
    def render_literal(value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def clean_sql(sql: str) -> str:
        lines = [re.sub(r"\s+", " ", line).rstrip() for line in sql.splitlines()]
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _metric_sql(metric: IRMetric) -> str:
        aggregation = metric.aggregation.upper()
        if aggregation == "COUNT" and metric.expression == "*":
            return "COUNT(*)"
        return f"{aggregation}({metric.expression})"

    def _filter_sql(self, item: IRFilter) -> str:
        expression = item.expression
        operator = item.operator
        value = item.value
        if operator == "equals":
            return f"{expression} = {self.render_literal(value)}"
        if operator == "not_equals":
            return f"{expression} <> {self.render_literal(value)}"
        if operator == "contains":
            return f"{expression} LIKE {self.render_literal('%' + str(value) + '%')}"
        if operator in {"in", "not_in"}:
            values = value if isinstance(value, list) else [value]
            rendered = ", ".join(self.render_literal(item_value) for item_value in values)
            return f"{expression} {'NOT IN' if operator == 'not_in' else 'IN'} ({rendered})"
        sql_operator = {
            "greater_than": ">",
            "greater_equal": ">=",
            "less_than": "<",
            "less_equal": "<=",
        }[operator]
        return f"{expression} {sql_operator} {self.render_literal(value)}"

    def _date_filter_sql(self, item: IRDateFilter) -> str:
        clauses = []
        if item.start_date:
            clauses.append(f"{item.date_expression} >= {self.render_literal(item.start_date)}")
        if item.end_date:
            clauses.append(f"{item.date_expression} < {self.render_literal(item.end_date)}")
        return " AND ".join(clauses)

    def _render_group_expression(self, query_ir: QueryIR, expression: str) -> str:
        if expression.startswith("DATE_GRAIN("):
            grain = self._grain_filter(query_ir)
            if grain:
                return self.render_date_grain(grain.date_expression, grain.date_grain or "month")
            return ""
        return expression

    @staticmethod
    def _grain_filter(query_ir: QueryIR) -> IRDateFilter | None:
        return next((item for item in query_ir.date_filters if item.filter_type == "grain"), None)

    @staticmethod
    def _first_date_filter(query_ir: QueryIR) -> IRDateFilter | None:
        return query_ir.date_filters[0] if query_ir.date_filters else None

    def _record_select_columns(self, query_ir: QueryIR) -> list[str]:
        if query_ir.dimensions:
            return [f"{item.expression} AS {item.alias}" for item in query_ir.dimensions]
        if query_ir.filters:
            return [item.expression for item in query_ir.filters]

        base_table = query_ir.base_table
        schema_tables = (
            (query_ir.metadata.get("validation_context") or {})
            .get("schema_context", {})
            .get("tables", {})
        )
        if base_table and base_table in schema_tables:
            columns = schema_tables[base_table].get("columns", {})
            safe = [
                f"{base_table}.{column}"
                for column in columns
                if not self._is_sensitive(column)
            ]
            return safe[:5] if safe else [f"{base_table}.rowid"]
        if base_table:
            return [f"{base_table}.rowid"]
        return ["NULL AS no_safe_columns"]

    @staticmethod
    def _is_sensitive(column: str) -> bool:
        name = column.lower()
        return any(marker in name for marker in SENSITIVE_MARKERS)
