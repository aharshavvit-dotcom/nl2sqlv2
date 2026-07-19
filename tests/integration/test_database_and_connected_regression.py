"""
Purpose: Verifies execution integration behaviour consolidated from fragmented test files.
Required because: Database connector and connected-database regression tests are one integration lane.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# Source: tests/test_03_database_connectors.py
"""Test 03: Database Connectors — connection config, SQLite connector, schema reader."""


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


# Source: tests/test_12_generic_postgres_schema_runtime.py
"""Full runtime checks for generic PostgreSQL-shaped schemas."""


import pytest

from generic_planner import SchemaProfile, TableIntentResolver
from inference.prediction_orchestrator import PredictionOrchestrator
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.schema_aware_mapper import SchemaAwareMapper
from inference.slot_resolver import SlotResolver
from tests.fixtures.generic_schema import GENERIC_POSTGRES_SCHEMA


NON_RETAIL_SCHEMA = {
    "dialect": "postgres",
    "tables": {
        **GENERIC_POSTGRES_SCHEMA["tables"],
        "berths": {"columns": [{"name": "id", "type": "integer", "is_primary_key": True}, {"name": "berth_name", "type": "text"}]},
        "vessels": {"columns": [{"name": "id", "type": "integer", "is_primary_key": True}, {"name": "vessel_name", "type": "text"}]},
        "terminals": {"columns": [{"name": "id", "type": "integer", "is_primary_key": True}, {"name": "terminal_name", "type": "text"}]},
        "service_orders": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "vessel_id", "type": "integer"},
                {"name": "terminal_id", "type": "integer"},
                {"name": "status", "type": "text"},
                {"name": "cost", "type": "numeric"},
                {"name": "created_at", "type": "timestamp"},
            ]
        },
    },
}


class ExplodingRetriever:
    def query(self, *_args, **_kwargs):
        raise AssertionError("retriever should be bypassed for direct schema-safe queries")


@pytest.mark.parametrize(
    ("question", "table", "extra_sql"),
    [
        ("list all users", "users", None),
        ("list all berth_masters", "berth_masters", None),
        ("list assignments", "assignments", None),
        ("count users", "users", "COUNT(*)"),
        ("show users where role is admin", "users", 'WHERE "users"."role" = \'admin\''),
    ],
)
def test_runtime_bypasses_models_for_generic_single_table_queries(
    question: str,
    table: str,
    extra_sql: str | None,
) -> None:
    result = PredictionOrchestrator().predict(
        question,
        schema=GENERIC_POSTGRES_SCHEMA,
        retriever=ExplodingRetriever(),
    )

    assert result.source_model == "generic_direct_planner"
    assert result.sql is not None
    assert result.validation["is_valid"], result.validation
    assert f'FROM "{table}"' in result.sql
    assert "JOIN" not in result.sql.upper()
    assert "password_hash" not in result.sql
    assert result.query_ir["base_table"] == table
    assert result.query_ir["required_tables"] == [table]
    assert result.query_ir["joins"] == []
    if extra_sql:
        assert extra_sql in result.sql


def test_explicit_join_language_is_not_directly_handled() -> None:
    direct = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA)).resolve("show assignments with user names")

    assert direct.handled is False
    assert "join" in (direct.reason or "")


def test_service_orders_does_not_enable_sample_retail_mapping() -> None:
    context = RuntimeSchemaContext(NON_RETAIL_SCHEMA)
    mapper = SchemaAwareMapper()

    assert mapper._deterministic_metric("sales", context) is None
    assert mapper._deterministic_metric("revenue", context) is None
    assert mapper._deterministic_filter("status", context) is None

    slots = SlotResolver().resolve_slots(
        "show total cost by terminal",
        {"template_id": "metric_by_dimension"},
        [],
        context,
    )["slots"]
    mapping = mapper.map_slots_to_schema(slots, context, template_id="metric_by_dimension")
    assert slots["metric"]["value"] == "cost"
    assert mapping.metric_table == "service_orders"
    assert mapping.metric_column == "cost"
    assert not mapping.mapping_reasons


def test_generic_join_base_table_has_no_retail_priority() -> None:
    assert RuntimeJoinPlanner.choose_base_table(None, "terminals", ["service_orders", "terminals"]) == "terminals"


# Source: tests/test_65_connected_db_regression_generator.py
from connected_db_testing.schema_case_generator import SchemaCaseGenerator
from tests.fixtures.generic_schema import generic_schema


def test_connected_db_regression_generator_creates_direct_and_join_cases() -> None:
    cases = SchemaCaseGenerator().generate_cases(generic_schema())
    case_ids = {case["case_id"] for case in cases}

    assert "list_users" in case_ids
    assert "count_users" in case_ids
    assert any(case["case_type"] == "explicit_join" for case in cases)
    list_users = next(case for case in cases if case["case_id"] == "list_users")
    assert "JOIN" in list_users["expected"]["must_not_include"]
    assert "password_hash" in list_users["expected"]["must_not_include"]


# Source: tests/test_66_connected_db_regression_runner.py
from connected_db_testing.generated_case_runner import ConnectedDBRegressionRunner
from tests.fixtures.generic_schema import generic_schema


class BadJoinModel:
    def predict(self, question: str, schema: dict, case: dict | None = None) -> dict:
        return {"sql": 'SELECT * FROM "users" JOIN "assignments" ON "assignments"."user_id" = "users"."id" LIMIT 100'}


def test_connected_db_regression_runner_catches_unnecessary_join_select_star_and_sensitive_leak() -> None:
    case = {
        "case_id": "list_users",
        "question": "list all users",
        "case_type": "direct_table_listing",
        "expected": {
            "base_table": "users",
            "must_include": ['FROM "users"'],
            "must_not_include": ["JOIN", "password_hash"],
            "must_have_limit": True,
            "must_not_select_star": True,
        },
    }

    report = ConnectedDBRegressionRunner().run([case], generic_schema(), BadJoinModel())

    assert report["summary"]["case_pass_rate"] == 0.0
    assert report["summary"]["unnecessary_join_count"] >= 1
    assert report["summary"]["select_star_count"] == 1


def test_connected_db_regression_runner_passes_generated_smoke_cases() -> None:
    from connected_db_testing.schema_case_generator import SchemaCaseGenerator

    cases = SchemaCaseGenerator().generate_cases(generic_schema(), max_tables=3)
    report = ConnectedDBRegressionRunner().run(cases, generic_schema())

    assert report["summary"]["case_pass_rate"] == 1.0


# Source: tests/test_134_database_integration.py
"""Unit and integration tests for database security, read-only policies, and statement limits."""


import sqlite3
import pandas as pd
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
        assert "readonly" in str(excinfo.value).lower() or "syntax" in str(excinfo.value).lower()


def test_sqlite_execution_metadata_records_duration_and_size(temp_sqlite_db):
    df = execute_select(temp_sqlite_db, "SELECT id FROM users LIMIT 1", max_rows=1)
    metadata = df.attrs["execution_metadata"]

    assert metadata["row_count"] == 1
    assert metadata["max_rows"] == 1
    assert metadata["duration_ms"] >= 0
    assert metadata["sql_sha256"]
    assert metadata["schema_fingerprint"]


def test_sqlite_stale_validation_result_is_revalidated(temp_sqlite_db):
    stale_validation = {"is_valid": True}

    with pytest.raises(ValueError) as excinfo:
        execute_select(temp_sqlite_db, "DROP TABLE users", validation_result=stale_validation)

    assert "SELECT" in str(excinfo.value) or "Only" in str(excinfo.value)


def test_sqlite_outer_limit_is_applied_before_fetch(temp_sqlite_db):
    with patch(
        "execution.query_executor.pd.read_sql_query",
        return_value=pd.DataFrame([{"id": 1}]),
    ) as read_sql:
        execute_select(temp_sqlite_db, "SELECT id FROM users LIMIT 1", max_rows=1)

    executed_sql = str(read_sql.call_args.args[0])
    assert "SELECT * FROM (" in executed_sql
    assert "LIMIT 1" in executed_sql


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
