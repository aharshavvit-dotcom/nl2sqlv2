from __future__ import annotations

import re
from typing import Any

from .query_ir_models import IRDateFilter, IRFilter, IRMetric, QueryIR


SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address", "dob", "birth_date", "credit_card", "api_key", "auth")
AUDIT_MARKERS = ("created_at", "updated_at", "deleted_at", "internal_id")


def quote_identifier(name: str | None, dialect: str = "sqlite") -> str:
    """Quote *name* as one SQL identifier token.

    Identifier punctuation is data, not syntax.  In particular, periods in
    schema column names (for example ``Rd.``) must never be split here.
    SQLite and Postgres both accept ANSI double-quoted identifiers; MySQL is
    also configured to accept them in the SQL emitted by this project.
    """
    del dialect  # Reserved for a future dialect with different quote rules.
    text = "" if name is None else str(name)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].replace('""', '"')
    elif len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        text = text[1:-1].replace("``", "`")
    return '"' + text.replace('"', '""') + '"'


class IRToSQLRenderer:
    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit

    @staticmethod
    def _quote_identifier(value: str | None, force: bool = False) -> str:
        del force  # Kept for compatibility with older callers.
        return quote_identifier(value)

    @staticmethod
    def _qualified_identifier(table: str | None, column: str | None) -> str:
        if column == "*":
            return f"{quote_identifier(table)}.*" if table else "*"
        if table:
            return f"{quote_identifier(table)}.{quote_identifier(column)}"
        return quote_identifier(column)

    @classmethod
    def _quote_expression(cls, expression: str | None) -> str:
        """Quote a simple IR identifier expression without tokenizing punctuation."""
        text = "" if expression is None else str(expression)
        if text == "*":
            return "*"
        if "." in text:
            table, column = text.split(".", 1)
            if table and column:
                return cls._qualified_identifier(table, column)
        return quote_identifier(text)

    @classmethod
    def _item_expression(cls, item: Any) -> str:
        table = getattr(item, "table", None)
        column = getattr(item, "column", None)
        if column:
            return cls._qualified_identifier(table, column)
        return cls._quote_expression(getattr(item, "expression", None))

    def render(self, query_ir: QueryIR | dict[str, Any], dialect: str | None = None) -> str:
        query_ir = self._coerce_query_ir(query_ir)
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

    @staticmethod
    def _coerce_query_ir(query_ir: QueryIR | dict[str, Any]) -> QueryIR:
        if isinstance(query_ir, QueryIR):
            return query_ir
        if isinstance(query_ir, dict):
            if hasattr(QueryIR, "model_validate"):
                return QueryIR.model_validate(query_ir)
            return QueryIR(**query_ir)
        raise TypeError(f"Expected QueryIR or dict, got {type(query_ir).__name__}")

    def render_select(self, query_ir: QueryIR, dialect: str = "sqlite") -> str:
        template_id = query_ir.template_id
        metric = query_ir.metrics[0] if query_ir.metrics else None
        dimension = query_ir.dimensions[0] if query_ir.dimensions else None
        force_quotes = bool(query_ir.metadata.get("force_quoted_identifiers"))

        if template_id == "count_records" or (query_ir.select_mode == "count" and not dimension):
            count_sql = self._metric_sql(metric) if metric else "COUNT(*)"
            count_alias = metric.alias if metric else "record_count"
            return f"SELECT\n  {count_sql} AS {quote_identifier(count_alias, dialect)}"
        if template_id == "count_by_dimension" and dimension:
            dim_expr = self._item_expression(dimension)
            count_sql = self._metric_sql(metric) if metric else "COUNT(*)"
            count_alias = metric.alias if metric else "record_count"
            return f"SELECT\n  {dim_expr} AS {quote_identifier(dimension.alias, dialect)},\n  {count_sql} AS {quote_identifier(count_alias, dialect)}"
        if template_id == "trend_by_date" and metric:
            grain = self._grain_filter(query_ir) or self._first_date_filter(query_ir)
            if grain:
                date_expr = self.render_date_grain(self._qualified_identifier(grain.date_table, grain.date_column), grain.date_grain or "month", dialect=dialect)
                return f"SELECT\n  {date_expr} AS {quote_identifier('period', dialect)},\n  {self._metric_sql(metric)} AS {quote_identifier(metric.alias, dialect)}"
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"} and metric and dimension:
            dim_expr = self._item_expression(dimension)
            return f"SELECT\n  {dim_expr} AS {quote_identifier(dimension.alias, dialect)},\n  {self._metric_sql(metric)} AS {quote_identifier(metric.alias, dialect)}"
        if template_id == "metric_summary" and metric:
            return f"SELECT\n  {self._metric_sql(metric)} AS {quote_identifier(metric.alias, dialect)}"

        record_columns = self._record_select_columns(query_ir)
        if force_quotes and query_ir.dimensions:
            record_columns = [f"{self._item_expression(item)} AS {quote_identifier(item.alias, dialect)}" for item in query_ir.dimensions]
        return "SELECT\n  " + ",\n  ".join(record_columns)

    def render_from(self, query_ir: QueryIR) -> str:
        force_quotes = bool(query_ir.metadata.get("force_quoted_identifiers"))
        return f"FROM {quote_identifier(query_ir.base_table)}" if query_ir.base_table else ""

    @staticmethod
    def render_joins(query_ir: QueryIR) -> str:
        lines = []
        for join in sorted(query_ir.joins, key=lambda item: item.path_order):
            join_type = (join.join_type or "INNER").upper()
            right = quote_identifier(join.right_table)
            condition = (
                f"{IRToSQLRenderer._qualified_identifier(join.left_table, join.left_column)} = "
                f"{IRToSQLRenderer._qualified_identifier(join.right_table, join.right_column)}"
            )
            lines.append(f"{join_type} JOIN {right}\n  ON {condition}")
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
        parts = [
            f"{quote_identifier(item.alias) if item.alias else IRToSQLRenderer._quote_expression(item.expression)} {item.direction}"
            for item in query_ir.order_by
        ]
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
        expr = IRToSQLRenderer._item_expression(metric)
        return f"{aggregation}({expr})"

    def _filter_sql(self, item: IRFilter, dialect: str = "sqlite", force_quotes: bool = False) -> str:
        expression = self._qualified_identifier(item.table, item.column)
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
        date_expr = self._qualified_identifier(item.date_table, item.date_column)
        if item.start_date:
            clauses.append(f"{date_expr} >= {self.render_literal(item.start_date)}")
        if item.end_date:
            clauses.append(f"{date_expr} < {self.render_literal(item.end_date)}")
        return " AND ".join(clauses)

    def _render_group_expression(self, query_ir: QueryIR, expression: str, dialect: str = "sqlite") -> str:
        if expression.startswith("DATE_GRAIN("):
            grain = self._grain_filter(query_ir)
            if grain:
                date_expr = self._qualified_identifier(grain.date_table, grain.date_column)
                return self.render_date_grain(date_expr, grain.date_grain or "month", dialect=dialect)
            return ""
        return self._quote_expression(expression)

    @staticmethod
    def _grain_filter(query_ir: QueryIR) -> IRDateFilter | None:
        return next((item for item in query_ir.date_filters if item.filter_type == "grain"), None)

    @staticmethod
    def _first_date_filter(query_ir: QueryIR) -> IRDateFilter | None:
        return query_ir.date_filters[0] if query_ir.date_filters else None

    def _record_select_columns(self, query_ir: QueryIR) -> list[str]:
        if query_ir.dimensions:
            return [f"{self._item_expression(item)} AS {quote_identifier(item.alias)}" for item in query_ir.dimensions]
        if query_ir.filters:
            return [self._qualified_identifier(item.table, item.column) for item in query_ir.filters]

        base_table = query_ir.base_table
        schema_tables = (
            (query_ir.metadata.get("validation_context") or {})
            .get("schema_context", {})
            .get("tables", {})
        )
        if base_table and base_table in schema_tables:
            columns = schema_tables[base_table].get("columns", {})
            safe = [
                self._qualified_identifier(base_table, column)
                for column in columns
                if not self._is_sensitive(column) and not self._is_audit(column)
            ]
            return safe[:4] if safe else [self._qualified_identifier(base_table, "rowid")]
        if base_table:
            return [self._qualified_identifier(base_table, "rowid")]
        return [f"NULL AS {quote_identifier('no_safe_columns')}"]

    @staticmethod
    def _is_sensitive(column: str) -> bool:
        name = column.lower()
        return any(marker in name for marker in SENSITIVE_MARKERS)

    @staticmethod
    def _is_audit(column: str) -> bool:
        name = column.lower()
        return any(name == marker or name.endswith(f"_{marker}") for marker in AUDIT_MARKERS)
