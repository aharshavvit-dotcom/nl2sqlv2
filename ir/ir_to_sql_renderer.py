from __future__ import annotations

import re
from typing import Any

from .query_ir_models import IRDateFilter, IRFilter, IRMetric, QueryIR


SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address", "dob", "birth_date", "credit_card", "api_key", "auth")


class IRToSQLRenderer:
    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit

    @staticmethod
    def _quote_identifier(value: str, force: bool = False) -> str:
        if not value:
            return ""
        if value == "*":
            return "*"
        if "." in value:
            parts = value.split(".")
            return ".".join(IRToSQLRenderer._quote_identifier(part, force=force) for part in parts)
        if (value.startswith('"') and value.endswith('"')) or (value.startswith('`') and value.endswith('`')):
            return value
        safe_regex = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if force and safe_regex.match(value):
            return f'"{value}"'
        if not safe_regex.match(value):
            return f'"{value}"'
        return value

    def render(self, query_ir: QueryIR, dialect: str | None = None) -> str:
        resolved_dialect = dialect or getattr(query_ir, 'dialect', 'sqlite') or 'sqlite'
        parts = [
            self.render_select(query_ir, dialect=resolved_dialect),
            self.render_from(query_ir),
            self.render_joins(query_ir),
            self.render_where(query_ir, dialect=resolved_dialect),
            self.render_group_by(query_ir, dialect=resolved_dialect),
            self.render_order_by(query_ir),
            self.render_limit(query_ir),
        ]
        return self.clean_sql("\n".join(part for part in parts if part))

    def render_select(self, query_ir: QueryIR, dialect: str = "sqlite") -> str:
        template_id = query_ir.template_id
        metric = query_ir.metrics[0] if query_ir.metrics else None
        dimension = query_ir.dimensions[0] if query_ir.dimensions else None
        force_quotes = bool(query_ir.metadata.get("force_quoted_identifiers"))

        if template_id == "count_records" or (query_ir.select_mode == "count" and not dimension):
            return "SELECT\n  COUNT(*) AS record_count"
        if template_id == "count_by_dimension" and dimension:
            dim_expr = self._quote_identifier(dimension.expression)
            return f"SELECT\n  {dim_expr} AS {dimension.alias},\n  COUNT(*) AS record_count"
        if template_id == "trend_by_date" and metric:
            grain = self._grain_filter(query_ir) or self._first_date_filter(query_ir)
            if grain:
                date_expr = self.render_date_grain(self._quote_identifier(grain.date_expression), grain.date_grain or "month", dialect=dialect)
                return f"SELECT\n  {date_expr} AS period,\n  {self._metric_sql(metric)} AS {metric.alias}"
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"} and metric and dimension:
            dim_expr = self._quote_identifier(dimension.expression)
            return f"SELECT\n  {dim_expr} AS {dimension.alias},\n  {self._metric_sql(metric)} AS {metric.alias}"
        if template_id == "metric_summary" and metric:
            return f"SELECT\n  {self._metric_sql(metric)} AS {metric.alias}"

        record_columns = self._record_select_columns(query_ir)
        if force_quotes and query_ir.dimensions:
            record_columns = [f"{self._quote_identifier(item.expression, force=True)} AS {item.alias}" for item in query_ir.dimensions]
        return "SELECT\n  " + ",\n  ".join(record_columns)

    def render_from(self, query_ir: QueryIR) -> str:
        force_quotes = bool(query_ir.metadata.get("force_quoted_identifiers"))
        return f"FROM {self._quote_identifier(query_ir.base_table, force=force_quotes)}" if query_ir.base_table else ""

    @staticmethod
    def render_joins(query_ir: QueryIR) -> str:
        lines = []
        for join in sorted(query_ir.joins, key=lambda item: item.path_order):
            join_type = (join.join_type or "INNER").upper()
            right = IRToSQLRenderer._quote_identifier(join.right_table)
            lines.append(f"{join_type} JOIN {right}\n  ON {join.condition}")
        return "\n".join(lines)

    def render_where(self, query_ir: QueryIR, dialect: str = "sqlite") -> str:
        force_quotes = bool(query_ir.metadata.get("force_quoted_identifiers"))
        clauses = [self._filter_sql(item, dialect=dialect, force_quotes=force_quotes) for item in query_ir.filters]
        clauses.extend(self._date_filter_sql(item) for item in query_ir.date_filters if item.filter_type != "grain")
        clauses = [clause for clause in clauses if clause]
        if not clauses:
            return ""
        return "WHERE " + "\n  AND ".join(clauses)

    def render_group_by(self, query_ir: QueryIR, dialect: str = "sqlite") -> str:
        expressions = [self._render_group_expression(query_ir, expression, dialect=dialect) for expression in query_ir.group_by]
        expressions = [expression for expression in expressions if expression]
        return "GROUP BY " + ", ".join(expressions) if expressions else ""

    @staticmethod
    def render_order_by(query_ir: QueryIR) -> str:
        if not query_ir.order_by:
            return ""
        parts = [f"{item.alias or IRToSQLRenderer._quote_identifier(item.expression)} {item.direction}" for item in query_ir.order_by]
        return "ORDER BY " + ", ".join(parts)

    def render_limit(self, query_ir: QueryIR) -> str:
        return f"LIMIT {min(max(int(query_ir.limit or 100), 1), self.max_limit)}"

    @staticmethod
    def render_date_grain(date_expression: str, grain: str, dialect: str = "sqlite") -> str:
        if dialect == "postgres":
            pg_grain = "year" if grain == "year" else "month"
            return f"TO_CHAR(DATE_TRUNC('{pg_grain}', {date_expression}), 'YYYY-MM')"
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
        expr = IRToSQLRenderer._quote_identifier(metric.expression)
        return f"{aggregation}({expr})"

    def _filter_sql(self, item: IRFilter, dialect: str = "sqlite", force_quotes: bool = False) -> str:
        expression = self._quote_identifier(item.expression, force=force_quotes)
        operator = item.operator
        value = item.value
        if operator == "equals":
            return f"{expression} = {self.render_literal(value)}"
        if operator == "not_equals":
            return f"{expression} <> {self.render_literal(value)}"
        if operator == "contains":
            if dialect == "postgres":
                return f"{expression} ILIKE {self.render_literal('%' + str(value) + '%')}"
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
        date_expr = self._quote_identifier(item.date_expression)
        if item.start_date:
            clauses.append(f"{date_expr} >= {self.render_literal(item.start_date)}")
        if item.end_date:
            clauses.append(f"{date_expr} < {self.render_literal(item.end_date)}")
        return " AND ".join(clauses)

    def _render_group_expression(self, query_ir: QueryIR, expression: str, dialect: str = "sqlite") -> str:
        if expression.startswith("DATE_GRAIN("):
            grain = self._grain_filter(query_ir)
            if grain:
                date_expr = self._quote_identifier(grain.date_expression)
                return self.render_date_grain(date_expr, grain.date_grain or "month", dialect=dialect)
            return ""
        return self._quote_identifier(expression)

    @staticmethod
    def _grain_filter(query_ir: QueryIR) -> IRDateFilter | None:
        return next((item for item in query_ir.date_filters if item.filter_type == "grain"), None)

    @staticmethod
    def _first_date_filter(query_ir: QueryIR) -> IRDateFilter | None:
        return query_ir.date_filters[0] if query_ir.date_filters else None

    def _record_select_columns(self, query_ir: QueryIR) -> list[str]:
        if query_ir.dimensions:
            return [f"{self._quote_identifier(item.expression)} AS {item.alias}" for item in query_ir.dimensions]
        if query_ir.filters:
            return [self._quote_identifier(item.expression) for item in query_ir.filters]

        base_table = query_ir.base_table
        schema_tables = (
            (query_ir.metadata.get("validation_context") or {})
            .get("schema_context", {})
            .get("tables", {})
        )
        if base_table and base_table in schema_tables:
            columns = schema_tables[base_table].get("columns", {})
            safe = [
                f"{self._quote_identifier(base_table)}.{self._quote_identifier(column)}"
                for column in columns
                if not self._is_sensitive(column)
            ]
            return safe[:5] if safe else [f"{self._quote_identifier(base_table)}.rowid"]
        if base_table:
            return [f"{self._quote_identifier(base_table)}.rowid"]
        return ["NULL AS no_safe_columns"]

    @staticmethod
    def _is_sensitive(column: str) -> bool:
        name = column.lower()
        return any(marker in name for marker in SENSITIVE_MARKERS)
