from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .schema import ForeignKeyInfo, SchemaGraph


@dataclass(frozen=True)
class JoinStep:
    table: str
    sql: str


class JoinResolver:
    def __init__(self, schema: SchemaGraph):
        self.schema = schema

    def resolve(self, base_table: str, required_tables: set[str]) -> list[JoinStep]:
        joins: list[JoinStep] = []
        joined = {base_table}

        for target in sorted(required_tables - joined):
            path = self._path_to_any(joined, target)
            if not path:
                raise ValueError(f"No join path from {sorted(joined)} to {target}")
            for fk, next_table in path:
                if next_table in joined:
                    continue
                joins.append(JoinStep(table=next_table, sql=self._join_sql(fk, next_table)))
                joined.add(next_table)
        return joins

    def _path_to_any(self, starts: set[str], target: str) -> list[tuple[ForeignKeyInfo, str]]:
        queue = deque([(start, []) for start in starts])
        seen = set(starts)
        while queue:
            table, path = queue.popleft()
            if table == target:
                return path
            for fk, neighbor in self.schema.neighbors(table):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, [*path, (fk, neighbor)]))
        return []

    @staticmethod
    def _join_sql(fk: ForeignKeyInfo, next_table: str) -> str:
        if next_table == fk.referred_table:
            return (
                f"JOIN {fk.referred_table} "
                f"ON {fk.table}.{fk.constrained_column} = "
                f"{fk.referred_table}.{fk.referred_column}"
            )
        return (
            f"JOIN {fk.table} "
            f"ON {fk.table}.{fk.constrained_column} = "
            f"{fk.referred_table}.{fk.referred_column}"
        )
