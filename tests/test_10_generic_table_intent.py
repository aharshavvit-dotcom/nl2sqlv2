"""Generic schema-first table intent planning."""

from __future__ import annotations

import pytest

from generic_planner import SchemaProfile, TableIntentResolver
from ir.ir_to_sql_renderer import IRToSQLRenderer
from validation.sql_validator import SQLValidator


GENERIC_POSTGRES_SCHEMA = {
    "dialect": "postgres",
    "database": "test_db",
    "schema_name": "public",
    "tables": {
        "users": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "name", "type": "text"},
                {"name": "role", "type": "text"},
                {"name": "created_at", "type": "timestamp"},
                {"name": "password_hash", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        "berth_masters": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "berth_name", "type": "text"},
                {"name": "berth_code", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        "assignments": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "user_id", "type": "integer", "is_foreign_key": True},
                {"name": "berth_id", "type": "integer", "is_foreign_key": True},
                {"name": "assigned_date", "type": "date"},
                {"name": "status", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [
                {"column": "user_id", "references_table": "users", "references_column": "id"},
                {"column": "berth_id", "references_table": "berth_masters", "references_column": "id"},
            ],
        },
    },
    "relationships": [
        {"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"},
        {"from_table": "assignments", "from_column": "berth_id", "to_table": "berth_masters", "to_column": "id"},
    ],
}


@pytest.mark.parametrize(
    ("question", "table"),
    [
        ("list all users", "users"),
        ("show users", "users"),
        ("display users", "users"),
        ("list all berth_masters", "berth_masters"),
        ("list all berth masters", "berth_masters"),
        ("show berth masters", "berth_masters"),
        ("list assignments", "assignments"),
    ],
)
def test_simple_table_questions_build_direct_queryir(question: str, table: str) -> None:
    result = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA)).resolve(question)

    assert result.handled is True
    assert result.intent == "show_records"
    query_ir = result.query_ir
    assert query_ir.base_table == table
    assert query_ir.required_tables == [table]
    assert query_ir.joins == []
    assert query_ir.metrics == []
    assert query_ir.metadata["source"] == "generic_direct_planner"

    sql = IRToSQLRenderer().render(query_ir, dialect="postgres")
    validation = SQLValidator().validate(sql, schema=GENERIC_POSTGRES_SCHEMA, dialect="postgres")

    assert validation["is_valid"], validation
    assert f'FROM "{table}"' in sql
    assert "JOIN" not in sql.upper()
    assert "LIMIT" in sql.upper()
    assert "SELECT *" not in sql.upper()
    assert "password_hash" not in sql


def test_users_default_projection_is_bounded_and_excludes_audit_columns() -> None:
    result = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA)).resolve("list all users")

    projected = [item.column for item in result.query_ir.dimensions]
    assert projected == ["id", "name", "role"]
    assert "created_at" not in projected
    assert "password_hash" not in projected
    assert result.query_ir.metadata["projection_mode"] == "list_all_records"
    assert result.query_ir.metadata["default_projection_used"] is True


def test_show_active_users_is_a_simple_filter_without_join() -> None:
    schema = {**GENERIC_POSTGRES_SCHEMA, "tables": {**GENERIC_POSTGRES_SCHEMA["tables"]}}
    schema["tables"]["users"] = {
        **GENERIC_POSTGRES_SCHEMA["tables"]["users"],
        "columns": [*GENERIC_POSTGRES_SCHEMA["tables"]["users"]["columns"], {"name": "status", "type": "text"}],
    }
    result = TableIntentResolver(SchemaProfile(schema)).resolve("show active users")

    assert result.handled is True
    assert result.intent == "simple_filter"
    assert result.query_ir.base_table == "users"
    assert result.query_ir.joins == []
    assert result.query_ir.filters[0].column == "status"
    assert result.query_ir.filters[0].value == "active"
