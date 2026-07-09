from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel
from inference.runtime_schema_context import RuntimeSchemaContext


class ProjectionResolution(BaseModel):
    projection_mode: str
    requested_columns: list[str]
    selected_columns: list[str]
    default_projection_used: bool
    excluded_sensitive_columns: list[str]
    confidence: float


class ProjectionResolver:
    def __init__(self, schema_context: RuntimeSchemaContext):
        self.schema_context = schema_context

    def resolve_projection(
        self,
        question: str,
        entity_table: str | None = None,
        metric_table: str | None = None,
    ) -> ProjectionResolution:
        q_lower = question.lower()
        requested = []
        selected = []
        excluded_sensitive = []
        mode = "list-all"
        default_used = False

        if "count" in q_lower or "how many" in q_lower:
            mode = "count-only"
            selected = ["*"]
            return ProjectionResolution(
                projection_mode=mode,
                requested_columns=[],
                selected_columns=selected,
                default_projection_used=False,
                excluded_sensitive_columns=[],
                confidence=0.95,
            )

        active_table = entity_table or metric_table or (self.schema_context.get_tables()[0] if self.schema_context.get_tables() else None)

        candidate_columns = []
        if active_table:
            candidate_columns = [f"{active_table}.{col}" for col in self.schema_context.get_table_columns(active_table)]
            for fk in self.schema_context.foreign_keys:
                ref_tbl = None
                if fk.get("child_table") == active_table:
                    ref_tbl = fk.get("parent_table")
                elif fk.get("parent_table") == active_table:
                    ref_tbl = fk.get("child_table")
                if ref_tbl:
                    candidate_columns.extend([f"{ref_tbl}.{col}" for col in self.schema_context.get_table_columns(ref_tbl)])

        for qualified in candidate_columns:
            table, column = qualified.split(".", 1)
            info = self.schema_context.column_info(table, column)
            col_phrase = column.lower().replace("_", " ")

            if col_phrase in q_lower:
                requested.append(qualified)
                if info.get("is_sensitive"):
                    excluded_sensitive.append(qualified)
                else:
                    selected.append(qualified)

        if selected:
            mode = "specific-column"
        else:
            default_used = True
            if active_table:
                cols = self.schema_context.get_table_columns(active_table)
                for col in cols:
                    qualified = f"{active_table}.{col}"
                    info = self.schema_context.column_info(active_table, col)
                    if info.get("is_sensitive"):
                        excluded_sensitive.append(qualified)
                    else:
                        selected.append(qualified)
                selected = selected[:4]

        return ProjectionResolution(
            projection_mode=mode,
            requested_columns=requested,
            selected_columns=selected,
            default_projection_used=default_used,
            excluded_sensitive_columns=excluded_sensitive,
            confidence=0.85,
        )
