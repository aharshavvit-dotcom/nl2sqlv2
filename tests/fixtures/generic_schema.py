"""Generic schema fixtures for planner, grounding and database tests.

Purpose: Provides non-retail schemas used by generic runtime and connected
database tests.
Required because: These fixtures are shared across consolidated test modules
and readiness audits.
"""

from __future__ import annotations


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


def generic_schema() -> dict:
    """Return the dict-column variant used by semantic-layer fixtures."""
    return {
        "dialect": "postgres",
        "tables": {
            "users": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "name": {"type": "text"},
                    "role": {"type": "text"},
                    "status": {"type": "text"},
                    "password_hash": {"type": "text"},
                    "created_at": {"type": "timestamp"},
                }
            },
            "berth_masters": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "berth_code": {"type": "text"},
                    "berth_name": {"type": "text"},
                    "status": {"type": "text"},
                }
            },
            "assignments": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "user_id": {"type": "integer"},
                    "berth_id": {"type": "integer"},
                    "status": {"type": "text"},
                    "assigned_date": {"type": "date"},
                }
            },
        },
        "relationships": [
            {"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"},
            {"from_table": "assignments", "from_column": "berth_id", "to_table": "berth_masters", "to_column": "id"},
        ],
    }
