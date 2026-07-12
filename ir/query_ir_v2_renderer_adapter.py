from __future__ import annotations

from typing import Any

from .ir_to_sql_renderer import IRToSQLRenderer
from .query_ir_migration import convert_v2_to_v1
from .query_ir_v2_models import QueryNode


class QueryIRV2RendererAdapter:
    """Render the v1-compatible QueryIR v2 subset through the existing v1 renderer."""

    def __init__(self, renderer: IRToSQLRenderer | None = None):
        self.renderer = renderer or IRToSQLRenderer()

    def render(self, query_ir: QueryNode | dict[str, Any], dialect: str | None = None) -> str:
        query_ir_v1 = convert_v2_to_v1(query_ir)
        return self.renderer.render(query_ir_v1, dialect=dialect)


__all__ = ["QueryIRV2RendererAdapter"]
