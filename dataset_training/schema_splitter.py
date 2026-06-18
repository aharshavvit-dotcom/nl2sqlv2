from __future__ import annotations

from collections import defaultdict
from typing import Any


class SchemaSplitter:
    def group_by_database(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in examples:
            grouped[str(row.get("db_id") or "__unknown_db__")].append(row)
        return dict(grouped)
