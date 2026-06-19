from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from generic_planner.generic_slot_resolver import (
    filter_sample_retail_physical_mappings,
    is_sample_retail_schema,
)

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
        template_id: str | None = None,
        semantic_profile: dict[str, Any] | None = None,
    ) -> SchemaMapping:
        metric_synonyms = normalize_section(metric_synonyms or {})
        dimension_synonyms = normalize_section(dimension_synonyms or {})
        metric_synonyms, dimension_synonyms = filter_sample_retail_physical_mappings(
            metric_synonyms,
            dimension_synonyms,
            schema_context.get_tables(),
        )
        slot_values = {key: value.get("value") if isinstance(value, dict) else value for key, value in slots.items()}
        mapping = SchemaMapping()
        retail_schema = is_sample_retail_schema(schema_context.get_tables())
        if retail_schema:
            mapping.mapping_reasons.append("sample_retail_physical_mappings_allowed:schema_signature_matched")
        dimension = slot_values.get("dimension")
        entity = slot_values.get("entity")
        semantic_mapper = _semantic_mapper(semantic_profile)

        entity_match = self._map_entity_semantic(str(entity), semantic_mapper) if entity and semantic_mapper else None
        if not entity_match:
            entity_match = self._map_entity(str(entity) if entity else None, schema_context, retail_schema)
        mapping.entity_table = entity_match.get("table")
        mapping.match_scores["entity"] = entity_match.get("score", 0.0)

        if template_id in {"show_records", "simple_filter"}:
            mapping.metric_name = None
            mapping.metric_table = None
            mapping.metric_column = None
            mapping.metric_expression = None
            mapping.metric_aggregation = None
            mapping.match_scores["metric"] = 1.0
        elif template_id == "count_records":
            mapping.metric_name = "record_count"
            mapping.metric_table = mapping.entity_table
            mapping.metric_column = None
            mapping.metric_expression = "*"
            mapping.metric_aggregation = "COUNT"
            mapping.metric_alias = "record_count"
            mapping.match_scores["metric"] = 1.0
        else:
            metric = str(slot_values.get("metric") or self._default_metric(metric_synonyms))
            metric_match = self._map_metric_semantic(metric, semantic_mapper, mapping.entity_table) if semantic_mapper else None
            if not metric_match:
                metric_match = self._map_metric(metric, schema_context, metric_synonyms, str(entity) if entity else None)
            mapping.metric_name = metric
            mapping.metric_table = metric_match.get("table")
            mapping.metric_column = metric_match.get("column")
            mapping.metric_aggregation = "COUNT" if metric in {"record_count", "order_count"} else ("AVG" if metric in {"average", "average_order_value"} else "SUM")
            mapping.match_scores["metric"] = metric_match.get("score", 0.0)
            mapping.warnings.extend(metric_match.get("warnings", []))

        if dimension:
            dimension_match = self._map_dimension_semantic(str(dimension), semantic_mapper, mapping.metric_table) if semantic_mapper else None
            if not dimension_match:
                dimension_match = self._map_dimension(str(dimension), schema_context, mapping.metric_table, dimension_synonyms)
            mapping.dimension_name = str(dimension)
            mapping.dimension_table = dimension_match.get("table")
            mapping.dimension_column = dimension_match.get("column")
            mapping.match_scores["dimension"] = dimension_match.get("score", 0.0)
            mapping.warnings.extend(dimension_match.get("warnings", []))

        filter_column = slot_values.get("filter_column")
        if filter_column:
            filter_match = self._map_dimension_semantic(str(filter_column), semantic_mapper, mapping.entity_table or mapping.metric_table) if semantic_mapper else None
            if not filter_match:
                filter_match = self._map_filter(
                    str(filter_column),
                    schema_context,
                    str(entity) if entity else None,
                    mapping.metric_table,
                    dimension_synonyms,
                )
            mapping.filter_table = filter_match.get("table")
            mapping.filter_column = filter_match.get("column")
            mapping.match_scores["filter"] = filter_match.get("score", 0.0)
            mapping.warnings.extend(filter_match.get("warnings", []))

        if template_id not in {"show_records", "simple_filter", "count_records"}:
            date_match = self._map_date_semantic("date", semantic_mapper, mapping.metric_table or mapping.entity_table) if semantic_mapper else None
            if not date_match:
                date_match = self._map_date(schema_context, preferred_table=mapping.metric_table or mapping.entity_table)
            mapping.date_table = date_match.get("table")
            mapping.date_column = date_match.get("column")
            mapping.match_scores["date"] = date_match.get("score", 0.0)
            if not date_match.get("column"):
                mapping.warnings.append("missing date column")
        return mapping

    @staticmethod
    def _map_entity_semantic(entity: str, mapper: Any) -> dict[str, Any] | None:
        result = mapper.map_table(entity)
        if result.get("matched") and result.get("score", 0.0) >= 0.70:
            return {"table": result.get("target"), "score": result.get("score")}
        return None

    @staticmethod
    def _map_metric_semantic(metric: str, mapper: Any, table: str | None = None) -> dict[str, Any] | None:
        result = mapper.map_metric(metric, table=table)
        if result.get("matched") and result.get("score", 0.0) >= 0.70:
            target = str(result.get("target"))
            item = mapper.profile.get("metrics", {}).get(target, {})
            return {"table": item.get("base_table"), "column": item.get("column"), "score": result.get("score"), "warnings": []}
        return None

    @staticmethod
    def _map_dimension_semantic(dimension: str, mapper: Any, table: str | None = None) -> dict[str, Any] | None:
        result = mapper.map_dimension(dimension, table=table)
        if result.get("matched") and result.get("score", 0.0) >= 0.70:
            target = str(result.get("target"))
            item = mapper.profile.get("dimensions", {}).get(target, {})
            return {"table": item.get("table"), "column": item.get("column"), "score": result.get("score"), "warnings": []}
        return None

    @staticmethod
    def _map_date_semantic(date_phrase: str, mapper: Any, table: str | None = None) -> dict[str, Any] | None:
        dates = mapper.profile.get("dates") or {}
        if table:
            for item in dates.values():
                if item.get("table") == table:
                    return {"table": item.get("table"), "column": item.get("column"), "score": item.get("confidence", 0.85)}
        result = mapper.map_date(date_phrase, table=table)
        if result.get("matched"):
            item = dates.get(str(result.get("target")), {})
            return {"table": item.get("table"), "column": item.get("column"), "score": result.get("score")}
        return None

    def _map_metric(
        self,
        metric: str,
        schema_context: RuntimeSchemaContext,
        metric_synonyms: dict[str, list[str]],
        preferred_table: str | None = None,
    ) -> dict[str, Any]:
        if metric in {"record_count", "order_count"}:
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

    def _map_filter(
        self,
        filter_name: str,
        schema_context: RuntimeSchemaContext,
        entity: str | None,
        metric_table: str | None,
        dimension_synonyms: dict[str, list[str]],
    ) -> dict[str, Any]:
        deterministic = self._deterministic_filter(filter_name, schema_context, entity)
        if deterministic:
            return deterministic
        return self._map_dimension(filter_name, schema_context, metric_table, dimension_synonyms)

    @staticmethod
    def _map_entity(
        entity: str | None,
        schema_context: RuntimeSchemaContext,
        retail_schema: bool = False,
    ) -> dict[str, Any]:
        if entity and schema_context.has_table(entity):
            return {"table": entity, "score": 1.0}
        aliases = ENTITY_TABLES.get(entity or "", []) if retail_schema else []
        if not aliases and entity:
            aliases = [entity]
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
        if not is_sample_retail_schema(schema_context.get_tables()):
            return None
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
    def _deterministic_filter(filter_name: str, schema_context: RuntimeSchemaContext, entity: str | None = None) -> dict[str, Any] | None:
        if not is_sample_retail_schema(schema_context.get_tables()):
            return None
        normalized = filter_name.lower().replace(" ", "_")
        entity_name = (entity or "").lower()
        preferred: dict[str, list[tuple[str, str, float]]] = {
            "status": [("orders", "status", 0.98)],
            "order_status": [("orders", "status", 0.98)],
            "category": [("products", "category", 0.98)],
            "product_category": [("products", "category", 0.98)],
            "region": [("customers", "region", 0.96), ("stores", "region", 0.92), ("sales_reps", "region", 0.9)],
            "customer_region": [("customers", "region", 0.98)],
            "store_region": [("stores", "region", 0.98)],
            "segment": [("customers", "segment", 0.95)],
            "customer_segment": [("customers", "segment", 0.98)],
            "brand": [("products", "brand", 0.95)],
        }
        candidates = preferred.get(normalized, [])
        if normalized == "region" and entity_name in {"store", "stores"}:
            candidates = [("stores", "region", 0.98), *candidates]
        for table, column, score in candidates:
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


def _semantic_mapper(semantic_profile: dict[str, Any] | None) -> Any | None:
    if not semantic_profile:
        return None
    try:
        from semantic_layer.semantic_mapper import SemanticMapper

        return SemanticMapper(semantic_profile)
    except Exception:
        return None
