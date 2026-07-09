from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.grounding.projection_resolver import ProjectionResolver


def test_count_projection_only():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    resolver = ProjectionResolver(ctx)
    res = resolver.resolve_projection("count the number of orders", entity_table="orders")
    assert res.projection_mode == "count-only"
    assert res.selected_columns == ["*"]


def test_specific_projection_only():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_table_columns.return_value = ["id", "customer_name", "secret_key"]

    def col_info(table, col):
        return {"is_sensitive": col == "secret_key"}
    ctx.column_info.side_effect = col_info
    ctx.foreign_keys = []

    resolver = ProjectionResolver(ctx)
    res = resolver.resolve_projection("show customer name of orders", entity_table="orders")
    assert res.projection_mode == "specific-column"
    assert "orders.customer_name" in res.selected_columns
    assert "orders.secret_key" not in res.selected_columns
