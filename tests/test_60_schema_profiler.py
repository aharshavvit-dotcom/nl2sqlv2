from __future__ import annotations

from semantic_layer.schema_profiler import SchemaProfiler


def generic_schema() -> dict:
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


def test_schema_profiler_detects_table_roles_and_sensitive_columns() -> None:
    profile = SchemaProfiler().profile(generic_schema())

    assert profile["tables"]["users"]["table_type"] == "entity"
    assert profile["tables"]["berth_masters"]["table_type"] == "lookup"
    assert profile["tables"]["assignments"]["table_type"] == "bridge"
    assert "password_hash" in profile["tables"]["users"]["sensitive_columns"]
    assert "password_hash" not in profile["tables"]["users"]["safe_columns"]
    assert "role" in profile["tables"]["users"]["likely_filters"]
    assert "created_at" in profile["tables"]["users"]["likely_dates"]
