from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from nl2sql_v1.schema import read_sqlite_schema, sqlite_url
from validation.sql_validator import SQLValidator


def execute_select(
    db_path: str | Path,
    sql: str,
    validation_result: dict[str, Any] | None = None,
    max_rows: int = 1000,
) -> pd.DataFrame:
    """Execute a validated SELECT against a SQLite database."""
    schema = read_sqlite_schema(db_path)
    validation = validation_result or SQLValidator().validate(sql, schema=schema, max_limit=max_rows)
    if not validation.get("is_valid", validation.get("ok", False)):
        issues = validation.get("issues") or [validation.get("message", "SQL validation failed")]
        raise ValueError("; ".join(str(issue) for issue in issues))

    engine = create_engine(sqlite_url(db_path), future=True)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")
        df = pd.read_sql_query(text(sql), conn)
    return df.head(max_rows) if len(df) > max_rows else df


def execute_query(config, sql: str, validation_result: dict[str, Any] | None = None, max_rows: int = 1000) -> pd.DataFrame:
    """Execute a validated SELECT against any supported database.

    Uses the ``db`` connector layer for PostgreSQL; falls back to
    ``execute_select`` for SQLite paths.
    """
    from db.connection_config import DatabaseConnectionConfig

    if isinstance(config, (str, Path)):
        return execute_select(config, sql, validation_result=validation_result, max_rows=max_rows)

    if isinstance(config, DatabaseConnectionConfig):
        if config.db_type == "sqlite":
            return execute_select(config.sqlite_path or "", sql, validation_result=validation_result, max_rows=max_rows)

        # PostgreSQL path
        from db.postgres_connector import PostgresConnector

        connector = PostgresConnector(config)
        result = connector.execute_readonly(sql, limit=max_rows)
        if result.get("error"):
            raise ValueError(result["error"])
        return pd.DataFrame(result["rows"], columns=result["columns"])

    raise TypeError(f"Unsupported config type: {type(config)}")
