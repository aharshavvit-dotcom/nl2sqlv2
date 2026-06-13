from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from .prediction_models import SchemaMapping
from .runtime_schema_context import RuntimeSchemaContext
from .synonym_loader import normalize_section


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
        metric_synonyms = normalize_section(metric_synonyms or {})
        dimension_synonyms = normalize_section(dimension_synonyms or {})
        slot_values = {key: value.get("value") if isinstance(value, dict) else value for key, value in slots.items()}
        mapping = SchemaMapping()
        metric = str(slot_values.get("metric") or self._default_metric(metric_synonyms))
        dimension = slot_values.get("dimension")
        entity = slot_values.get("entity")

        metric_match = self._map_metric(metric, schema_context, metric_synonyms, str(entity) if entity else None)
        mapping.metric_name = metric
        mapping.metric_table = metric_match.get("table")
        mapping.metric_column = metric_match.get("column")
        mapping.metric_aggregation = "COUNT" if metric == "order_count" else ("AVG" if metric in {"average", "average_order_value"} else "SUM")
        mapping.match_scores["metric"] = metric_match.get("score", 0.0)
        mapping.warnings.extend(metric_match.get("warnings", []))

        if dimension:
            dimension_match = self._map_dimension(str(dimension), schema_context, mapping.metric_table, dimension_synonyms)
            mapping.dimension_name = str(dimension)
            mapping.dimension_table = dimension_match.get("table")
            mapping.dimension_column = dimension_match.get("column")
            mapping.match_scores["dimension"] = dimension_match.get("score", 0.0)
            mapping.warnings.extend(dimension_match.get("warnings", []))

        filter_column = slot_values.get("filter_column")
        if filter_column:
            filter_match = self._map_dimension(str(filter_column), schema_context, mapping.metric_table, dimension_synonyms)
            mapping.filter_table = filter_match.get("table")
            mapping.filter_column = filter_match.get("column")
            mapping.match_scores["filter"] = filter_match.get("score", 0.0)
            mapping.warnings.extend(filter_match.get("warnings", []))

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

    def _map_metric(
        self,
        metric: str,
        schema_context: RuntimeSchemaContext,
        metric_synonyms: dict[str, list[str]],
        preferred_table: str | None = None,
    ) -> dict[str, Any]:
        if metric == "order_count":
            if preferred_table and schema_context.has_table(preferred_table):
                pk = self._primary_key_or_id(schema_context, preferred_table)
                if pk:
                    return {"table": preferred_table, "column": pk, "score": 0.95, "warnings": []}
            candidates = []
            for table in schema_context.get_tables():
                pk = self._primary_key_or_id(schema_context, table)
                if pk:
                    aliases = self._aliases(metric, metric_synonyms)
                    score = max(fuzz.WRatio(alias, f"{table}.{pk}") for alias in aliases) / 100
                    candidates.append((score, table, pk))
            if candidates:
                score, table, column = max(candidates, key=lambda item: item[0])
                return {"table": table, "column": column, "score": round(max(score, 0.75), 4), "warnings": []}
        deterministic = self._deterministic_metric(metric, schema_context)
        if deterministic:
            return deterministic
        aliases = self._aliases(metric, metric_synonyms)
        candidates = []
        for qualified in schema_context.get_numeric_columns():
            table, column = qualified.split(".", 1)
            score = max(max(fuzz.WRatio(alias, column), fuzz.WRatio(alias, qualified)) for alias in aliases) / 100
            candidates.append((min(score, 1.0), table, column))
        if not candidates:
            return {"table": None, "column": None, "score": 0.0, "warnings": ["low metric match"]}
        score, table, column = max(candidates, key=lambda item: item[0])
        warnings = [] if score >= 0.55 else ["low metric match"]
        return {"table": table, "column": column, "score": round(score, 4), "warnings": warnings}

    def _map_dimension(
        self,
        dimension: str,
        schema_context: RuntimeSchemaContext,
        metric_table: str | None,
        dimension_synonyms: dict[str, list[str]],
    ) -> dict[str, Any]:
        if dimension in {"month", "year"}:
            date = self._map_date(schema_context, preferred_table=metric_table)
            return date if date.get("column") else {"table": None, "column": None, "score": 0.0, "warnings": ["missing date column"]}
        aliases = self._aliases(dimension, dimension_synonyms)
        candidates = []
        for qualified in [*schema_context.get_text_columns(), *schema_context.get_numeric_columns()]:
            table, column = qualified.split(".", 1)
            info = schema_context.column_info(table, column)
            if info["is_sensitive"]:
                continue
            alias_names = [alias.lower() for alias in aliases]
            column_name = column.lower()
            score = max(max(fuzz.WRatio(alias, column), fuzz.WRatio(alias, qualified)) for alias in aliases) / 100
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
        score, table, column = max(candidates, key=lambda item: item[0])
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
            candidates.append((min(score, 1.0), table))
        if not candidates:
            return {"table": None, "score": 0.0}
        score, table = max(candidates, key=lambda item: item[0])
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
        score, table, column = max(candidates, key=lambda item: item[0])
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

    @staticmethod
    def _deterministic_metric(metric: str, schema_context: RuntimeSchemaContext) -> dict[str, Any] | None:
        normalized = metric.lower().replace(" ", "_")
        preferred = {
            "sales": ("orders", "amount", 0.98),
            "revenue": ("orders", "amount", 0.98),
            "total_sales": ("orders", "amount", 0.98),
            "average_order_value": ("orders", "amount", 0.95),
            "quantity": ("order_items", "quantity", 0.95),
            "profit": ("order_items", "profit", 0.95),
            "discount": ("order_items", "discount", 0.95),
        }
        target = preferred.get(normalized)
        if not target:
            return None
        table, column, score = target
        if schema_context.has_column(table, column):
            return {"table": table, "column": column, "score": score, "warnings": []}
        return None

    @staticmethod
    def _aliases(key: str, synonyms: dict[str, list[str]]) -> list[str]:
        base = [key, key.replace("_", " ")]
        return [*base, *synonyms.get(key, [])]

    @staticmethod
    def _default_metric(metric_synonyms: dict[str, list[str]]) -> str:
        return next(iter(metric_synonyms), "metric")
