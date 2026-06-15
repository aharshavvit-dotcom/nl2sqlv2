"""Test 02: SQL Validation — SQLValidator, safe preview, dialect handling."""

from __future__ import annotations

from validation.sql_validator import SQLValidator
from app.safe_preview import build_safe_preview_sql


class TestSQLValidator:
    def test_rejects_sensitive_column(self) -> None:
        schema = {"tables": {"customers": {"columns": {"customer_id": {}, "email": {}}}}}
        result = SQLValidator().validate("SELECT customers.email FROM customers LIMIT 10", schema=schema)
        assert not result["is_valid"]
        assert not result["checks"]["no_sensitive_columns"]

    def test_accepts_valid_select(self) -> None:
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        result = SQLValidator().validate("SELECT order_id, amount FROM orders LIMIT 10", schema=schema)
        assert result["is_valid"]

    def test_rejects_insert(self) -> None:
        result = SQLValidator().validate("INSERT INTO orders VALUES (1, 100)")
        assert not result["is_valid"]

    def test_rejects_drop(self) -> None:
        result = SQLValidator().validate("DROP TABLE orders")
        assert not result["is_valid"]

    def test_rejects_delete(self) -> None:
        result = SQLValidator().validate("DELETE FROM orders WHERE order_id = 1")
        assert not result["is_valid"]

    def test_sqlite_dialect(self) -> None:
        result = SQLValidator().validate("SELECT order_id FROM orders LIMIT 10", dialect="sqlite")
        assert result["is_valid"]

    def test_postgres_dialect(self) -> None:
        result = SQLValidator().validate("SELECT order_id FROM orders LIMIT 10", dialect="postgres")
        assert result["is_valid"]

    def test_rejects_dob_sensitive_column(self) -> None:
        schema = {"tables": {"users": {"columns": {"user_id": {}, "dob": {}}}}}
        result = SQLValidator().validate("SELECT users.dob FROM users LIMIT 10", schema=schema)
        assert not result["is_valid"]

    def test_rejects_credit_card_column(self) -> None:
        schema = {"tables": {"payments": {"columns": {"id": {}, "credit_card": {}}}}}
        result = SQLValidator().validate("SELECT payments.credit_card FROM payments LIMIT 10", schema=schema)
        assert not result["is_valid"]


class TestSafePreviewSQL:
    def test_builds_safe_preview(self) -> None:
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        sql = build_safe_preview_sql("orders", schema)
        assert sql is not None
        assert "order_id" in sql
        assert "LIMIT" in sql

    def test_excludes_sensitive_columns(self) -> None:
        schema = {"tables": {"users": {"columns": {"user_id": {}, "email": {}, "phone": {}}}}}
        sql = build_safe_preview_sql("users", schema)
        assert sql is not None
        assert "email" not in sql
        assert "phone" not in sql
