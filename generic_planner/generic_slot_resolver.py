from __future__ import annotations

from typing import Any

from .schema_text_normalizer import table_name_variants


RETAIL_TABLES = {"orders", "customers", "products", "order_items"}


def is_sample_retail_schema(schema_tables: list[str] | set[str]) -> bool:
    return RETAIL_TABLES.issubset({str(table).lower() for table in schema_tables})


def runtime_schema_generated_synonyms(schema_tables: list[str] | set[str]) -> dict[str, list[str]]:
    return {
        str(table): sorted(table_name_variants(str(table)))
        for table in schema_tables
    }


def filter_sample_retail_physical_mappings(
    metric_synonyms: dict[str, Any] | None,
    dimension_synonyms: dict[str, Any] | None,
    schema_tables: list[str] | set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop bundled retail physical mappings for unrelated connected schemas."""
    if is_sample_retail_schema(schema_tables):
        return metric_synonyms or {}, dimension_synonyms or {}
    return {}, {}
