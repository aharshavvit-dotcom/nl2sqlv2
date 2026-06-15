"""Test 03: Database Connectors — connection config, SQLite connector, schema reader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from db.connection_config import DatabaseConnectionConfig, safe_config_summary
from db.dialect import get_sqlglot_dialect, SUPPORTED_DIALECTS
from db.schema_reader import read_database_schema, schema_dict_to_graph, schema_summary


ROOT = Path(__file__).resolve().parents[1]


class TestDatabaseConnectionConfig:
    def test_sqlite_config(self) -> None:
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path="/tmp/test.db")
        assert config.dialect == "sqlite"
        assert config.db_type == "sqlite"

    def test_postgres_config(self) -> None:
        config = DatabaseConnectionConfig(db_type="postgres", host="localhost",
                                          port=5432, database="testdb",
                                          username="user", password="secret")
        assert config.dialect == "postgres"
        assert config.schema_name == "public"

    def test_postgresql_normalizes_to_postgres(self) -> None:
        config = DatabaseConnectionConfig(db_type="postgresql")
        assert config.db_type == "postgres"

    def test_unsupported_db_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            DatabaseConnectionConfig(db_type="mysql")


class TestSafeConfigSummary:
    def test_masks_password(self) -> None:
        config = DatabaseConnectionConfig(db_type="postgres", host="h", port=5432,
                                          database="db", username="u", password="supersecret")
        summary = safe_config_summary(config)
        assert summary["password"] == "***"
        assert "supersecret" not in str(summary)

    def test_no_password_shows_none(self) -> None:
        config = DatabaseConnectionConfig(db_type="postgres", host="h", port=5432,
                                          database="db", username="u")
        summary = safe_config_summary(config)
        assert summary["password"] is None

    def test_sqlite_summary(self) -> None:
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path="/tmp/test.db")
        summary = safe_config_summary(config)
        assert summary["db_type"] == "sqlite"
        assert "sqlite_path" in summary


class TestDialect:
    def test_supported_dialects(self) -> None:
        assert "sqlite" in SUPPORTED_DIALECTS
        assert "postgres" in SUPPORTED_DIALECTS

    def test_get_sqlglot_dialect_postgres_aliases(self) -> None:
        assert get_sqlglot_dialect("postgresql") == "postgres"
        assert get_sqlglot_dialect("pg") == "postgres"

    def test_get_sqlglot_dialect_defaults_to_sqlite(self) -> None:
        assert get_sqlglot_dialect("") == "sqlite"
        assert get_sqlglot_dialect("unknown") == "sqlite"


class TestSQLiteConnector:
    @pytest.fixture()
    def sample_db(self, tmp_path: Path) -> Path:
        from scripts.create_sample_db import build_database
        db_path = tmp_path / "test.db"
        build_database(db_path)
        return db_path

    def test_test_connection(self, sample_db: Path) -> None:
        from db.sqlite_connector import SQLiteConnector
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        connector = SQLiteConnector(config)
        success, msg = connector.test_connection()
        assert success
        assert "Connected" in msg

    def test_read_schema(self, sample_db: Path) -> None:
        from db.sqlite_connector import SQLiteConnector
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        connector = SQLiteConnector(config)
        schema = connector.read_schema()
        assert schema["dialect"] == "sqlite"
        assert "orders" in schema["tables"]
        assert len(schema["tables"]) >= 3

    def test_execute_readonly(self, sample_db: Path) -> None:
        from db.sqlite_connector import SQLiteConnector
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        connector = SQLiteConnector(config)
        result = connector.execute_readonly("SELECT order_id FROM orders LIMIT 5")
        assert "error" not in result
        assert len(result["rows"]) <= 5

    def test_get_dialect(self, sample_db: Path) -> None:
        from db.sqlite_connector import SQLiteConnector
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        connector = SQLiteConnector(config)
        assert connector.get_dialect() == "sqlite"


class TestSchemaReader:
    @pytest.fixture()
    def sample_db(self, tmp_path: Path) -> Path:
        from scripts.create_sample_db import build_database
        db_path = tmp_path / "test.db"
        build_database(db_path)
        return db_path

    def test_read_database_schema_sqlite(self, sample_db: Path) -> None:
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        result = read_database_schema(config)
        assert result["dialect"] == "sqlite"
        assert "tables" in result

    def test_schema_dict_to_graph(self, sample_db: Path) -> None:
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        schema_dict = read_database_schema(config)
        graph = schema_dict_to_graph(schema_dict)
        assert "orders" in graph.tables
        assert "amount" in graph.tables["orders"].columns

    def test_schema_summary(self, sample_db: Path) -> None:
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(sample_db))
        schema_dict = read_database_schema(config)
        summary = schema_summary(schema_dict)
        assert summary["table_count"] >= 3
        assert summary["column_count"] > 0
        assert "orders" in summary["tables"]
