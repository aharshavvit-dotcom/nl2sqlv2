from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.grounding.dimension_resolver import DimensionResolver


def test_grouping_dimension():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_columns.return_value = ["orders.status", "orders.id"]
    ctx.column_info.return_value = {"is_sensitive": False}

    resolver = DimensionResolver(ctx)
    res = resolver.resolve_dimension("count orders grouped by status", "status", active_table="orders")
    assert res["role"] == "grouping"
    assert res["column"] == "status"
    assert res["table"] == "orders"


def test_dimension_unreachable_penalty():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_columns.return_value = ["customers.region", "suppliers.region"]
    ctx.column_info.return_value = {"is_sensitive": False}
    ctx.foreign_keys = []

    resolver = DimensionResolver(ctx)
    res = resolver.resolve_dimension("sales by region", "region", active_table="customers")
    assert res["table"] == "customers"
    assert res["column"] == "region"
