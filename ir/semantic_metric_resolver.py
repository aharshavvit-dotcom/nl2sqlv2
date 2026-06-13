from __future__ import annotations

from typing import Any


PRODUCT_DIMENSIONS = {"product", "products", "item", "items", "sku", "skus"}
REVENUE_METRICS = {"sales", "revenue", "total_sales", "total revenue", "total sales"}
PRODUCT_REVENUE_WARNING = (
    "Product-level revenue could not be resolved safely because item-level quantity/price columns were not found."
)


class SemanticMetricResolver:
    def resolve_metric_expression(
        self,
        metric_name: str,
        dimension_name: str | None,
        schema_context: dict[str, Any] | object,
        current_metric_table: str | None,
        current_metric_column: str | None,
    ) -> dict[str, Any]:
        metric = (metric_name or "").lower().replace(" ", "_")
        dimension = (dimension_name or "").lower().replace(" ", "_")
        if metric not in REVENUE_METRICS or dimension not in PRODUCT_DIMENSIONS:
            return {
                "resolved": False,
                "metric_table": current_metric_table,
                "metric_column": current_metric_column,
                "metric_expression": None,
                "metric_aggregation": None,
                "metric_alias": None,
                "required_tables": [],
                "warnings": [],
                "semantic_grain_risk": False,
            }

        if self._has_column(schema_context, "order_items", "quantity") and self._has_column(schema_context, "order_items", "price"):
            return {
                "resolved": True,
                "metric_table": "order_items",
                "metric_column": None,
                "metric_expression": "order_items.quantity * order_items.price",
                "metric_aggregation": "SUM",
                "metric_alias": "revenue",
                "required_tables": ["order_items", "products"],
                "warnings": [],
                "semantic_grain_risk": False,
            }

        return {
            "resolved": False,
            "metric_table": current_metric_table,
            "metric_column": current_metric_column,
            "metric_expression": None,
            "metric_aggregation": None,
            "metric_alias": "revenue",
            "required_tables": ["products"],
            "warnings": [PRODUCT_REVENUE_WARNING],
            "semantic_grain_risk": True,
        }

    @staticmethod
    def _has_column(schema_context: dict[str, Any] | object, table: str, column: str) -> bool:
        if hasattr(schema_context, "has_column"):
            return bool(schema_context.has_column(table, column))
        tables = (schema_context.get("tables", schema_context) if isinstance(schema_context, dict) else {}) or {}
        info = tables.get(table) or {}
        columns = info.get("columns", info) if isinstance(info, dict) else {}
        return column in columns
