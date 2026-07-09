"""Unit and integration tests for database security, read-only policies, and statement limits."""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from execution.query_executor import execute_select
from validation.sql_validator import SQLValidator
from db.postgres_connector import PostgresConnector
from db.connection_config import DatabaseConnectionConfig


@pytest.fixture
def temp_sqlite_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES ('Alice')")
    conn.commit()
    conn.close()
    return db_path


def test_sqlite_read_only_policy_blocks_writes(temp_sqlite_db):
    sql_write = "INSERT INTO users (name) VALUES ('Bob')"
    
    # Try executing – should fail either due to SQLValidator blocking INSERT
    # or SQLite PRAGMA query_only blocking write.
    with pytest.raises(ValueError) as excinfo:
        execute_select(temp_sqlite_db, sql_write)
        
    assert "SELECT" in str(excinfo.value) or "readonly" in str(excinfo.value)


def test_sqlite_query_only_pragma_enforced(temp_sqlite_db):
    # If someone tries to bypass SQLValidator using schema updates or similar:
    # We can mock SQLValidator validation to return passed=True, and verify SQLite itself blocks the write.
    sql_write = "INSERT INTO users (name) VALUES ('Bob')"
    
    with patch("validation.sql_validator.SQLValidator.validate", return_value={"is_valid": True}):
        with pytest.raises(Exception) as excinfo:
            execute_select(temp_sqlite_db, sql_write)
        # Should raise "attempt to write a readonly database"
        assert "readonly" in str(excinfo.value)


def test_sql_validator_blocks_multiple_statements():
    validator = SQLValidator()
    
    # Multi-statement injection attempt
    sql = "SELECT * FROM users; DROP TABLE users;"
    res = validator.validate(sql)
    
    assert res["is_valid"] is False
    assert "Only one SQL statement is allowed." in res["issues"]


def test_postgres_read_only_and_timeout_enforced():
    config = DatabaseConnectionConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="test_db",
        username="user",
        password="pwd",
    )
    
    connector = PostgresConnector(config)
    
    # Mock sqlalchemy engine and connect context manager
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    
    with patch("db.postgres_connector.create_engine", return_value=mock_engine), \
         patch("validation.sql_validator.SQLValidator.validate", return_value={"is_valid": True}):
         
        # Execute readonly
        connector.execute_readonly("SELECT 1", limit=10)
        
        # Verify read-only transaction and statement timeout are set on connection
        calls = [c[0][0].text for c in mock_conn.execute.call_args_list]
        assert any("SET TRANSACTION READ ONLY" in call for call in calls)
        assert any("SET LOCAL statement_timeout" in call for call in calls)
