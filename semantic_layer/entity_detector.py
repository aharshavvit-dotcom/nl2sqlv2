from __future__ import annotations

from typing import Any

from .schema_profiler import classify_table, normalize_schema, table_aliases_for


class EntityDetector:
    def detect(self, schema: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_schema(schema)
        relationships = normalized["relationships"]
        entities: dict[str, dict[str, Any]] = {}
        for table, info in normalized["tables"].items():
            fk_count = sum(1 for rel in relationships if rel.get("from_table") == table)
            table_type = profile.get("tables", {}).get(table, {}).get("table_type") or classify_table(table, info.get("columns", []), fk_count, [])
            if table_type in {"entity", "lookup", "bridge"}:
                entities[table] = {"table": table, "table_type": table_type, "aliases": table_aliases_for(table), "confidence": 0.85}
        return {"entities": entities}
