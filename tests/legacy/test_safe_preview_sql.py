"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from app.safe_preview import build_safe_preview_sql
from validation.sql_validator import SQLValidator


SCHEMA = {
    "tables": {
        "customers": {
            "columns": {
                "customer_id": {},
                "customer_name": {},
                "email": {},
                "phone": {},
                "region": {},
                "secret_token": {},
            }
        }
    }
}


def test_safe_preview_sql_excludes_select_star_and_sensitive_columns() -> None:
    sql = build_safe_preview_sql("customers", SCHEMA, max_columns=10, limit=20)

    assert sql is not None
    assert "SELECT *" not in sql
    assert "email" not in sql
    assert "phone" not in sql
    assert "secret_token" not in sql
    assert "LIMIT 20" in sql
    assert SQLValidator().validate(sql, schema=SCHEMA)["is_valid"]


def test_safe_preview_sql_returns_none_when_no_safe_columns() -> None:
    schema = {"tables": {"secrets": {"columns": {"email": {}, "password_hash": {}, "token": {}}}}}

    assert build_safe_preview_sql("secrets", schema) is None
