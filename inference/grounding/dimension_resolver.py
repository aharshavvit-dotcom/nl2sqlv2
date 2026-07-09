from __future__ import annotations

from typing import Any
from rapidfuzz import fuzz
from inference.runtime_schema_context import RuntimeSchemaContext


class DimensionResolver:
    def __init__(self, schema_context: RuntimeSchemaContext):
        self.schema_context = schema_context

    def resolve_dimension(
        self,
        question: str,
        dimension_phrase: str,
        active_table: str | None = None,
    ) -> dict[str, Any]:
        phrase = dimension_phrase.lower().strip()

        role = "display"
        q_lower = question.lower()
        if f"by {phrase}" in q_lower or "group by" in q_lower or "monthly" in q_lower or "yearly" in q_lower:
            role = "grouping"
        elif "order by" in q_lower or "sort by" in q_lower:
            role = "ordering"

        candidates = []
        for qualified in self.schema_context.get_columns():
            table, column = qualified.split(".", 1)
            info = self.schema_context.column_info(table, column)
            if info.get("is_sensitive"):
                continue

            col_phrase = column.lower().replace("_", " ")
            score = fuzz.ratio(phrase, col_phrase) / 100.0

            if active_table and table == active_table:
                score += 0.15
            elif active_table:
                if not self._is_reachable(table, active_table):
                    score -= 0.40

            candidates.append({
                "table": table,
                "column": column,
                "score": round(max(0.0, min(1.0, score)), 4),
                "role": role,
                "method": "fuzzy",
            })

        candidates.sort(key=lambda x: -x["score"])

        if candidates and candidates[0]["score"] >= 0.40:
            return candidates[0]

        return {
            "table": active_table,
            "column": None,
            "score": 0.0,
            "role": role,
            "method": "fallback",
        }

    def _is_reachable(self, t1: str, t2: str) -> bool:
        if t1 == t2:
            return True
        visited = {t1}
        queue = [t1]
        while queue:
            node = queue.pop(0)
            if node == t2:
                return True
            for fk in self.schema_context.foreign_keys:
                nbr = None
                if fk.get("child_table") == node:
                    nbr = fk.get("parent_table")
                elif fk.get("parent_table") == node:
                    nbr = fk.get("child_table")
                if nbr and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return False
