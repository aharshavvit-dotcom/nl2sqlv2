"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from validation.sql_validator import SQLValidator


SCHEMA = {
    "tables": {
        "orders": {"columns": {"order_id": {}, "amount": {}, "status": {}}},
        "customers": {"columns": {"customer_id": {}, "customer_name": {}, "email": {}}},
    }
}


def test_central_sql_validator_accepts_safe_select() -> None:
    result = SQLValidator().validate(
        """
        SELECT orders.status, COUNT(*) AS record_count
        FROM orders
        GROUP BY orders.status
        ORDER BY record_count DESC
        LIMIT 100
        """,
        schema=SCHEMA,
    )

    assert result["is_valid"]
    assert result["checks"]["no_select_star"]


def test_central_sql_validator_rejects_select_star_and_mutation() -> None:
    star = SQLValidator().validate("SELECT * FROM orders LIMIT 10", schema=SCHEMA)
    table_star = SQLValidator().validate("SELECT orders.* FROM orders LIMIT 10", schema=SCHEMA)
    mutation = SQLValidator().validate("SELECT order_id FROM orders; DROP TABLE orders", schema=SCHEMA)

    assert not star["is_valid"]
    assert not star["checks"]["no_select_star"]
    assert not table_star["is_valid"]
    assert not table_star["checks"]["no_select_star"]
    assert not mutation["is_valid"]
    assert not mutation["checks"]["single_statement"]


def test_central_sql_validator_rejects_sensitive_and_missing_limit() -> None:
    sensitive = SQLValidator().validate("SELECT email FROM customers LIMIT 10", schema=SCHEMA)
    missing_limit = SQLValidator().validate("SELECT order_id FROM orders", schema=SCHEMA)

    assert not sensitive["is_valid"]
    assert not sensitive["checks"]["no_sensitive_columns"]
    assert not missing_limit["is_valid"]
    assert not missing_limit["checks"]["limit_present"]
