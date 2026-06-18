from __future__ import annotations

from typing import Any

from .schema_profiler import column_aliases_for, is_date_column, normalize_schema


class DateColumnDetector:
    def detect(self, schema: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_schema(schema)
        dates: dict[str, dict[str, Any]] = {}
        for table, info in normalized["tables"].items():
            for column in info.get("columns", []):
                if not is_date_column(column):
                    continue
                key = f"{table}.{column['name']}"
                dates[key] = {
                    "table": table,
                    "column": column["name"],
                    "aliases": column_aliases_for(column["name"]),
                    "confidence": 0.9,
                }
        return {"dates": dates}
