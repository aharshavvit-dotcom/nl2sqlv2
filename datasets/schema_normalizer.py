from __future__ import annotations

import re
from typing import Any

from .models import DatabaseSchema


class SchemaNormalizer:
    @staticmethod
    def normalize_table_name(name: str) -> str:
        return SchemaNormalizer._normalize_identifier(name)

    @staticmethod
    def normalize_column_name(name: str) -> str:
        return SchemaNormalizer._normalize_identifier(name)

    @staticmethod
    def _normalize_identifier(name: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
        return normalized.strip("_")

    @staticmethod
    def serialize_schema(schema: DatabaseSchema) -> str:
        table_chunks: list[str] = []
        for table_name, table_info in schema.tables.items():
            columns = SchemaNormalizer._columns_from_table_info(table_info)
            table_chunks.append(f"{table_name}({', '.join(columns)})")
        return "tables: " + "; ".join(table_chunks)

    @staticmethod
    def build_schema_registry(
        examples: list[Any],
        schemas: dict[str, DatabaseSchema],
    ) -> list[dict[str, Any]]:
        used_db_ids = {example.db_id for example in examples}
        registry: list[dict[str, Any]] = []
        for db_id in sorted(used_db_ids):
            schema = schemas.get(db_id)
            if not schema:
                continue
            serialized = schema.serialized_schema or SchemaNormalizer.serialize_schema(schema)
            registry.append(
                {
                    "db_id": schema.db_id,
                    "dataset_name": schema.dataset_name,
                    "db_path": schema.db_path,
                    "tables": schema.tables,
                    "foreign_keys": schema.foreign_keys,
                    "primary_keys": schema.primary_keys,
                    "serialized_schema": serialized,
                }
            )
        return registry

    @staticmethod
    def _columns_from_table_info(table_info: Any) -> list[str]:
        if isinstance(table_info, dict):
            if isinstance(table_info.get("columns"), list):
                columns = table_info["columns"]
                return [str(item.get("name", item)) if isinstance(item, dict) else str(item) for item in columns]
            if isinstance(table_info.get("columns"), dict):
                return [str(key) for key in table_info["columns"]]
        return []
