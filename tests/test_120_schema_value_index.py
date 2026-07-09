from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.grounding.schema_value_index import SchemaValueIndex, ValueIndexMode


def test_sensitive_column_explicitly_excluded():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_tables.return_value = ["users"]
    ctx.get_table_columns.return_value = ["password", "username"]
    ctx.get_columns.return_value = ["users.password", "users.username"]

    def col_info(table, column):
        if column == "password":
            return {"is_sensitive": True, "sample_values": ["secret123"]}
        return {"is_sensitive": False, "sample_values": ["john_doe"]}
    ctx.column_info.side_effect = col_info

    idx = SchemaValueIndex(ctx, mode=ValueIndexMode.APPROVED_DOMAIN_VALUES)
    assert "secret123" not in idx.index
    assert "john doe" in idx.index


def test_sensitive_column_inferred_from_name():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_tables.return_value = ["users"]
    ctx.get_table_columns.return_value = ["ssn", "name"]
    ctx.get_columns.return_value = ["users.ssn", "users.name"]

    def col_info(table, column):
        if column == "ssn":
            return {"is_sensitive": False, "sample_values": ["123-456-7890"]}
        return {"is_sensitive": False, "sample_values": ["Jane"]}
    ctx.column_info.side_effect = col_info

    idx = SchemaValueIndex(ctx, mode=ValueIndexMode.APPROVED_DOMAIN_VALUES)
    assert "123 456 7890" not in idx.index
    assert "jane" in idx.index


def test_disabled_mode_performs_no_lookup():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_tables.return_value = ["users"]
    ctx.get_columns.return_value = ["users.name"]
    ctx.column_info.return_value = {"is_sensitive": False, "sample_values": ["Jane"]}

    idx = SchemaValueIndex(ctx, mode=ValueIndexMode.DISABLED)
    assert not idx.index
    assert idx.lookup_value("Jane") == []
