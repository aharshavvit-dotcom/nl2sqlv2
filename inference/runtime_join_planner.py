from __future__ import annotations

from collections import deque

from .prediction_models import JoinPlan
from .runtime_schema_context import RuntimeSchemaContext


class RuntimeJoinPlanner:
    def plan_joins(
        self,
        schema_context: RuntimeSchemaContext,
        base_table: str,
        required_tables: list[str],
    ) -> JoinPlan:
        required = [table for table in dict.fromkeys(required_tables) if table and table != base_table]
        if not required:
            return JoinPlan(base_table=base_table, required_tables=[base_table], join_clause="", join_steps=[], confidence=1.0)

        joined = {base_table}
        steps: list[dict[str, str]] = []
        warnings: list[str] = []
        for target in required:
            path = self._shortest_path(schema_context, joined, target)
            if not path:
                warnings.append(f"no join path from {sorted(joined)} to {target}")
                continue
            for edge in path:
                next_table = edge["neighbor"]
                if next_table in joined:
                    continue
                steps.append(
                    {
                        **edge,
                        "join_type": "INNER",
                        "condition": f"{edge['from_table']}.{edge['from_column']} = {edge['to_table']}.{edge['to_column']}",
                    }
                )
                joined.add(next_table)

        join_sql = "\n".join(self._join_sql(edge) for edge in steps)
        confidence = 1.0 if not warnings else 0.35
        return JoinPlan(
            base_table=base_table,
            required_tables=[base_table, *required],
            join_clause=join_sql,
            join_steps=steps,
            confidence=confidence,
            warnings=warnings,
        )

    @staticmethod
    def choose_base_table(metric_table: str | None, entity_table: str | None, required_tables: list[str]) -> str:
        if metric_table:
            return metric_table
        for preferred in ["orders", "sales", "transactions", "invoices", "order_items"]:
            if preferred in required_tables:
                return preferred
        return entity_table or required_tables[0]

    @staticmethod
    def _shortest_path(
        schema_context: RuntimeSchemaContext,
        starts: set[str],
        target: str,
    ) -> list[dict[str, str]]:
        queue = deque([(start, []) for start in starts])
        seen = set(starts)
        while queue:
            table, path = queue.popleft()
            if table == target:
                return path
            for edge in schema_context.relationships.get(table, []):
                neighbor = edge["neighbor"]
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, [*path, {**edge, "current": table}]))
        return []

    @staticmethod
    def _join_sql(edge: dict[str, str]) -> str:
        current = edge.get("current")
        if current == edge["to_table"]:
            join_table = edge["from_table"]
        else:
            join_table = edge["to_table"]
        left_table = edge["from_table"]
        right_table = edge["to_table"]
        return (
            f"JOIN {join_table}\n"
            f"  ON {left_table}.{edge['from_column']} = {right_table}.{edge['to_column']}"
        )
