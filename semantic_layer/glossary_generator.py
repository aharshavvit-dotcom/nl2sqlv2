from __future__ import annotations

from typing import Any

from .schema_profiler import column_aliases_for, normalize_schema, table_aliases_for


class GlossaryGenerator:
    def generate(self, schema: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_schema(schema)
        table_aliases = {}
        column_aliases = {}
        for table, info in normalized["tables"].items():
            table_aliases[table] = sorted(set((profile.get("tables", {}).get(table, {}).get("aliases") or []) + table_aliases_for(table)))
            for column in info.get("columns", []):
                key = f"{table}.{column['name']}"
                column_aliases[key] = column_aliases_for(column["name"])
        return {"tables": table_aliases, "columns": column_aliases}
