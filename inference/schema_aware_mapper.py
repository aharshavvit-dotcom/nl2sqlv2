from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from .prediction_models import SchemaMapping
from .runtime_schema_context import RuntimeSchemaContext


METRIC_COLUMNS = {
    "revenue": ["amount", "revenue", "sales", "total", "total_amount", "price", "value", "net_sales"],
    "sales": ["amount", "revenue", "sales", "total", "total_amount", "price", "value", "net_sales"],
    "quantity": ["quantity", "qty", "units", "count"],
    "order_count": ["order_id", "id"],
    "profit": ["profit", "margin"],
    "discount": ["discount", "markdown"],
    "average_order_value": ["amount", "order_amount", "order_value", "revenue", "sales", "total"],
}

DIMENSION_COLUMNS = {
    "customer": ["customer_name", "customer", "client_name", "buyer_name", "name"],
    "product": ["product_name", "product", "item_name", "sku_name", "name"],
    "region": ["region", "area", "territory", "zone"],
    "status": ["status", "state", "condition"],
    "category": ["category", "department"],
    "store": ["store_name", "store", "branch", "shop"],
    "brand": ["brand", "manufacturer"],
    "city": ["city", "town"],
    "state": ["state", "province"],
    "customer_segment": ["segment", "customer_segment", "tier"],
    "sales_rep": ["rep_name", "sales_rep", "representative", "salesperson"],
}

ENTITY_TABLES = {
    "orders": ["orders", "sales", "transactions", "invoices"],
    "customers": ["customers", "clients", "buyers"],
    "products": ["products", "items", "skus"],
    "order_items": ["order_items", "items", "line_items"],
}


class SchemaAwareMapper:
    def map_slots_to_schema(
        self,
        slots: dict[str, Any],
        schema_context: RuntimeSchemaContext,
        metric_synonyms: dict[str, Any] | None = None,
        dimension_synonyms: dict[str, Any] | None = None,
    ) -> SchemaMapping:
        slot_values = {key: value.get("value") if isinstance(value, dict) else value for key, value in slots.items()}
        mapping = SchemaMapping()
        metric = str(slot_values.get("metric") or "revenue")
        dimension = slot_values.get("dimension")
        entity = slot_values.get("entity")

        metric_match = self._map_metric(metric, schema_context)
        mapping.metric_name = metric
        mapping.metric_table = metric_match.get("table")
        mapping.metric_column = metric_match.get("column")
        mapping.metric_aggregation = "COUNT" if metric == "order_count" else ("AVG" if metric in {"average", "average_order_value"} else "SUM")
        mapping.match_scores["metric"] = metric_match.get("score", 0.0)
        mapping.warnings.extend(metric_match.get("warnings", []))

        if dimension:
            dimension_match = self._map_dimension(str(dimension), schema_context, mapping.metric_table)
            mapping.dimension_name = str(dimension)
            mapping.dimension_table = dimension_match.get("table")
            mapping.dimension_column = dimension_match.get("column")
            mapping.match_scores["dimension"] = dimension_match.get("score", 0.0)
            mapping.warnings.extend(dimension_match.get("warnings", []))

        entity_match = self._map_entity(str(entity) if entity else None, schema_context)
        mapping.entity_table = entity_match.get("table")
        mapping.match_scores["entity"] = entity_match.get("score", 0.0)

        date_match = self._map_date(schema_context, preferred_table=mapping.metric_table or mapping.entity_table)
        mapping.date_table = date_match.get("table")
        mapping.date_column = date_match.get("column")
        mapping.match_scores["date"] = date_match.get("score", 0.0)
        if not date_match.get("column"):
            mapping.warnings.append("missing date column")
        return mapping

    def _map_metric(self, metric: str, schema_context: RuntimeSchemaContext) -> dict[str, Any]:
        if metric == "order_count":
            for table in ["orders", "transactions", "invoices"]:
                if schema_context.has_table(table):
                    pk = self._primary_key_or_id(schema_context, table)
                    if pk:
                        return {"table": table, "column": pk, "score": 0.95, "warnings": []}
        aliases = METRIC_COLUMNS.get(metric, [metric])
        candidates = []
        for qualified in schema_context.get_numeric_columns():
            table, column = qualified.split(".", 1)
            score = max(fuzz.WRatio(alias, column) for alias in aliases) / 100
            if table in {"orders", "sales", "transactions", "invoices", "order_items"}:
                score += 0.08
            candidates.append((min(score, 1.0), table, column))
        if not candidates:
            return {"table": None, "column": None, "score": 0.0, "warnings": ["low metric match"]}
        score, table, column = max(candidates)
        warnings = [] if score >= 0.55 else ["low metric match"]
        return {"table": table, "column": column, "score": round(score, 4), "warnings": warnings}

    def _map_dimension(
        self,
        dimension: str,
        schema_context: RuntimeSchemaContext,
        metric_table: str | None,
    ) -> dict[str, Any]:
        if dimension in {"month", "year"}:
            date = self._map_date(schema_context, preferred_table=metric_table)
            return date if date.get("column") else {"table": None, "column": None, "score": 0.0, "warnings": ["missing date column"]}
        aliases = DIMENSION_COLUMNS.get(dimension, [dimension])
        candidates = []
        for qualified in [*schema_context.get_text_columns(), *schema_context.get_numeric_columns()]:
            table, column = qualified.split(".", 1)
            info = schema_context.column_info(table, column)
            if info["is_sensitive"]:
                continue
            alias_names = [alias.lower() for alias in aliases]
            column_name = column.lower()
            score = max(fuzz.WRatio(alias, column) for alias in aliases) / 100
            if column_name == dimension.lower():
                score += 0.25
            if column_name in alias_names:
                score += 0.15
            if metric_table and table == metric_table:
                score += 0.1
            if dimension in table.lower():
                score += 0.18
            if info["is_id"] and any(non_id in " ".join(aliases) for non_id in ["name", "customer", "product"]):
                score -= 0.2
            candidates.append((max(0.0, score), table, column))
        if not candidates:
            return {"table": None, "column": None, "score": 0.0, "warnings": ["low dimension match"]}
        score, table, column = max(candidates)
        warnings = [] if score >= 0.5 else ["low dimension match"]
        return {"table": table, "column": column, "score": round(min(score, 1.0), 4), "warnings": warnings}

    @staticmethod
    def _map_entity(entity: str | None, schema_context: RuntimeSchemaContext) -> dict[str, Any]:
        if entity and schema_context.has_table(entity):
            return {"table": entity, "score": 1.0}
        aliases = ENTITY_TABLES.get(entity or "", [entity] if entity else [])
        candidates = []
        for table in schema_context.get_tables():
            score = max([fuzz.WRatio(alias, table) for alias in aliases] or [0]) / 100
            if table in {"orders", "sales", "transactions", "invoices"}:
                score += 0.1
            candidates.append((min(score, 1.0), table))
        if not candidates:
            return {"table": None, "score": 0.0}
        score, table = max(candidates)
        return {"table": table, "score": round(score, 4)}

    @staticmethod
    def _map_date(schema_context: RuntimeSchemaContext, preferred_table: str | None = None) -> dict[str, Any]:
        date_columns = schema_context.get_date_columns()
        if not date_columns:
            return {"table": None, "column": None, "score": 0.0}
        candidates = []
        for qualified in date_columns:
            table, column = qualified.split(".", 1)
            score = 0.75
            if preferred_table and table == preferred_table:
                score += 0.15
            if column in {"order_date", "created_date", "transaction_date", "invoice_date", "sale_date"}:
                score += 0.1
            candidates.append((min(score, 1.0), table, column))
        score, table, column = max(candidates)
        return {"table": table, "column": column, "score": round(score, 4)}

    @staticmethod
    def _primary_key_or_id(schema_context: RuntimeSchemaContext, table: str) -> str | None:
        for column in schema_context.get_table_columns(table):
            info = schema_context.column_info(table, column)
            if info["primary_key"]:
                return column
        for column in schema_context.get_table_columns(table):
            if column.endswith("_id") or column == "id":
                return column
        return None
