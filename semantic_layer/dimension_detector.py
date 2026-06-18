from __future__ import annotations

from typing import Any

from .schema_profiler import column_aliases_for, is_dimension_column, normalize_schema


class DimensionDetector:
    def detect(self, schema: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_schema(schema)
        dimensions: dict[str, dict[str, Any]] = {}
        for table, info in normalized["tables"].items():
            for column in info.get("columns", []):
                if not is_dimension_column(column):
                    continue
                key = f"{table}.{column['name']}"
                dimensions[key] = {
                    "table": table,
                    "column": column["name"],
                    "aliases": column_aliases_for(column["name"]),
                    "confidence": 0.85,
                }
        return {"dimensions": dimensions}
