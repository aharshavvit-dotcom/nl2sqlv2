from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any

from nl2sql_v1.schema import SchemaGraph

from .query_ir_models import IRValidationIssue, IRValidationResult, QueryIR


SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address", "dob", "birth_date", "credit_card", "api_key", "auth")
METRIC_INTENTS = {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "trend_by_date"}
DIMENSION_INTENTS = {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"}
EXECUTABLE_INTENTS = {
    *METRIC_INTENTS,
    *DIMENSION_INTENTS,
    "count_records",
    "count_by_dimension",
    "simple_filter",
    "show_records",
}


class IRValidator:
    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit

    def validate(self, query_ir: QueryIR, schema: dict[str, Any] | SchemaGraph | None = None) -> IRValidationResult:
        issues: list[IRValidationIssue] = []
        schema_tables = self._schema_tables(schema)

        if not query_ir.intent:
            self._issue(issues, "error", "missing_intent", "QueryIR has no intent.")
        if query_ir.intent not in EXECUTABLE_INTENTS:
            self._issue(issues, "error", "unsupported_intent", f"Unsupported intent: {query_ir.intent}.")
        if not query_ir.base_table:
            self._issue(issues, "error", "missing_base_table", "Executable QueryIR requires a base table.")
        if query_ir.limit <= 0:
            self._issue(issues, "error", "invalid_limit", "Limit must be greater than zero.")
        if query_ir.limit > self.max_limit:
            self._issue(issues, "error", "limit_too_large", f"Limit must be <= {self.max_limit}.")
        if query_ir.template_id in METRIC_INTENTS and not query_ir.metrics:
            self._issue(issues, "error", "missing_metric", "Metric intent requires at least one metric.")
        if query_ir.template_id in DIMENSION_INTENTS and not query_ir.dimensions:
            self._issue(issues, "error", "missing_dimension", "By-dimension intent requires a dimension.")
        if (
            query_ir.metadata.get("source") == "generic_direct_planner"
            and query_ir.template_id in {"show_records", "simple_filter"}
            and not query_ir.dimensions
        ):
            self._issue(
                issues,
                "error",
                "no_safe_select_columns",
                "No safe non-sensitive columns are available for this direct table query.",
            )

        for date_filter in query_ir.date_filters:
            if not date_filter.date_table or not date_filter.date_column:
                self._issue(issues, "error", "missing_date_column", "Date filter requires a mapped date column.")

        if any(metric.expression == "*" and metric.aggregation.upper() != "COUNT" for metric in query_ir.metrics):
            self._issue(issues, "error", "select_star", "Only COUNT(*) may use '*' in QueryIR.")

        self._validate_semantics(issues, query_ir, schema_tables)

        if schema_tables:
            self._validate_tables(issues, query_ir, schema_tables)
            self._validate_columns(issues, query_ir, schema_tables)
            self._validate_joins(issues, query_ir, schema_tables)

        for table, column in self._referenced_columns(query_ir):
            if self._is_sensitive(column):
                self._issue(issues, "error", "sensitive_column", f"Sensitive column is referenced: {table}.{column}.")

        errors = [issue.message for issue in issues if issue.severity == "error"]
        warnings = [issue.message for issue in issues if issue.severity == "warning"]
        return IRValidationResult(is_valid=not errors, issues=issues, errors=errors, warnings=warnings)

    def _validate_tables(self, issues: list[IRValidationIssue], query_ir: QueryIR, schema_tables: dict[str, set[str]]) -> None:
        for table in query_ir.required_tables:
            if table and table not in schema_tables:
                self._issue(issues, "error", "unknown_table", f"Unknown required table: {table}.")
        if query_ir.base_table and query_ir.base_table not in schema_tables:
            self._issue(issues, "error", "unknown_base_table", f"Unknown base table: {query_ir.base_table}.")

    def _validate_columns(self, issues: list[IRValidationIssue], query_ir: QueryIR, schema_tables: dict[str, set[str]]) -> None:
        for table, column in self._referenced_columns(query_ir):
            if column == "*":
                continue
            if table not in schema_tables:
                self._issue(issues, "error", "unknown_column_table", f"Column table does not exist: {table}.{column}.")
                continue
            if column not in schema_tables[table]:
                self._issue(issues, "error", "unknown_column", f"Unknown column: {table}.{column}.")

    def _validate_joins(self, issues: list[IRValidationIssue], query_ir: QueryIR, schema_tables: dict[str, set[str]]) -> None:
        for join in query_ir.joins:
            for table, column in [(join.left_table, join.left_column), (join.right_table, join.right_column)]:
                if table not in schema_tables:
                    self._issue(issues, "error", "unknown_join_table", f"Join table does not exist: {table}.")
                elif column not in schema_tables[table]:
                    self._issue(issues, "error", "unknown_join_column", f"Join column does not exist: {table}.{column}.")

    @staticmethod
    def _referenced_columns(query_ir: QueryIR) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        refs.extend((metric.table, metric.column) for metric in query_ir.metrics if metric.table and metric.column)
        for metric in query_ir.metrics:
            refs.extend(IRValidator._expression_refs(metric.expression))
        refs.extend((dimension.table, dimension.column) for dimension in query_ir.dimensions)
        for dimension in query_ir.dimensions:
            refs.extend(IRValidator._expression_refs(dimension.expression))
        refs.extend((item.table, item.column) for item in query_ir.filters)
        for item in query_ir.filters:
            refs.extend(IRValidator._expression_refs(item.expression))
        refs.extend((item.date_table, item.date_column) for item in query_ir.date_filters)
        return [(table, column) for table, column in refs if table and column]

    def _validate_semantics(
        self,
        issues: list[IRValidationIssue],
        query_ir: QueryIR,
        schema_tables: dict[str, set[str]],
    ) -> None:
        dimension_names = {dimension.name.lower().replace(" ", "_") for dimension in query_ir.dimensions}
        metric_names = {metric.name.lower().replace(" ", "_") for metric in query_ir.metrics}
        asks_product_revenue = bool(dimension_names & {"product", "products", "item", "items", "sku"}) and bool(
            metric_names & {"sales", "revenue", "total_sales"}
        )
        if not asks_product_revenue:
            return
        metric_expression = " ".join(metric.expression for metric in query_ir.metrics).lower()
        safe_expression = (
            "order_items.quantity" in metric_expression
            and "order_items.price" in metric_expression
            and "orders.amount" not in metric_expression
        )
        if safe_expression:
            return
        if schema_tables and {"order_items", "products"}.issubset(schema_tables):
            self._issue(
                issues,
                "warning",
                "semantic_grain_risk",
                "Product-level revenue should use item-level quantity/price rather than order-level amount.",
            )

    @staticmethod
    def _expression_refs(expression: str | None) -> list[tuple[str, str]]:
        if not expression or expression == "*":
            return []
        return [(table, column) for table, column in re.findall(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", expression)]

    @staticmethod
    def _schema_tables(schema: dict[str, Any] | SchemaGraph | None) -> dict[str, set[str]]:
        if schema is None:
            return {}
        if isinstance(schema, SchemaGraph):
            return {table: set(info.columns) for table, info in schema.tables.items()}
        raw_tables = schema.get("tables", schema)
        normalized: dict[str, set[str]] = {}
        for table, info in raw_tables.items():
            if isinstance(info, dict):
                columns = info.get("columns", info)
            else:
                columns = getattr(info, "columns", {})
            if isinstance(columns, dict):
                normalized[str(table)] = {str(column) for column in columns}
            else:
                values = []
                for column in columns:
                    raw = asdict(column) if is_dataclass(column) else column
                    values.append(str(raw.get("name", raw)) if isinstance(raw, dict) else str(raw))
                normalized[str(table)] = set(values)
        return normalized

    @staticmethod
    def _is_sensitive(column: str) -> bool:
        name = column.lower()
        return any(marker in name for marker in SENSITIVE_MARKERS)

    @staticmethod
    def _issue(
        issues: list[IRValidationIssue],
        severity: str,
        issue_type: str,
        message: str,
        suggested_action: str | None = None,
    ) -> None:
        issues.append(
            IRValidationIssue(
                severity=severity,  # type: ignore[arg-type]
                issue_type=issue_type,
                message=message,
                suggested_action=suggested_action,
            )
        )
