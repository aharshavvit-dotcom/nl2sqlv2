from __future__ import annotations

from validation.sql_validator import SQLValidator


def test_sql_validator_rejects_sensitive_column() -> None:
    schema = {"tables": {"customers": {"columns": {"customer_id": {}, "email": {}}}}}

    result = SQLValidator().validate("SELECT customers.email FROM customers LIMIT 10", schema=schema)

    assert not result["is_valid"]
    assert not result["checks"]["no_sensitive_columns"]
