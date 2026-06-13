from __future__ import annotations

from validation.sql_validator import SQLValidator


SCHEMA = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}, "status": {}}}}}


def test_central_sql_validator_accepts_safe_select() -> None:
    result = SQLValidator().validate("SELECT orders.status FROM orders LIMIT 10", schema=SCHEMA)

    assert result["is_valid"]
    assert result["checks"]["no_select_star"]


def test_central_sql_validator_rejects_select_star_and_mutation() -> None:
    star = SQLValidator().validate("SELECT * FROM orders LIMIT 10", schema=SCHEMA)
    mutation = SQLValidator().validate("DELETE FROM orders", schema=SCHEMA)

    assert not star["is_valid"]
    assert not star["checks"]["no_select_star"]
    assert not mutation["is_valid"]
    assert not mutation["checks"]["select_only"]
