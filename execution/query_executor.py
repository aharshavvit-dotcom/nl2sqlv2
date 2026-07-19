from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
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
    timeout_ms: int = 5000,
) -> pd.DataFrame:
    """Execute a validated SELECT against a SQLite database."""
    schema = read_sqlite_schema(db_path)
    max_rows = max(1, int(max_rows or 1))
    schema_fingerprint = _schema_fingerprint(schema)
    if validation_result and _validation_matches(validation_result, sql, schema_fingerprint, max_rows):
        validation = dict(validation_result)
    else:
        validation = SQLValidator().validate(sql, schema=schema, max_limit=max_rows)
    validation["execution_provenance"] = _validation_provenance(sql, schema_fingerprint, max_rows)
    if not validation.get("is_valid", validation.get("ok", False)):
        issues = validation.get("issues") or [validation.get("message", "SQL validation failed")]
        raise ValueError("; ".join(str(issue) for issue in issues))

    bounded_sql = _outer_limited_select(sql, max_rows)
    engine = create_engine(sqlite_url(db_path), future=True)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")
        raw_connection = _sqlite_driver_connection(conn)
        start = time.monotonic()
        if raw_connection is not None and hasattr(raw_connection, "set_progress_handler"):
            def _abort_if_slow() -> int:
                return 1 if (time.monotonic() - start) * 1000 > timeout_ms else 0

            raw_connection.set_progress_handler(_abort_if_slow, 1000)
        try:
            df = pd.read_sql_query(text(bounded_sql), conn)
        except Exception as exc:
            if (time.monotonic() - start) * 1000 > timeout_ms:
                raise TimeoutError(f"SQLite query exceeded timeout_ms={timeout_ms}") from exc
            raise
        finally:
            if raw_connection is not None and hasattr(raw_connection, "set_progress_handler"):
                raw_connection.set_progress_handler(None, 0)
    duration_ms = round((time.monotonic() - start) * 1000, 3)
    df.attrs["execution_metadata"] = {
        "duration_ms": duration_ms,
        "row_count": int(len(df)),
        "max_rows": max_rows,
        "sql_sha256": validation["execution_provenance"]["sql_sha256"],
        "bounded_sql_sha256": hashlib.sha256(bounded_sql.strip().encode("utf-8")).hexdigest(),
        "schema_fingerprint": schema_fingerprint,
    }
    return df


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


def _outer_limited_select(sql: str, max_rows: int) -> str:
    stripped = sql.strip().rstrip(";")
    return f"SELECT * FROM ({stripped}) AS nl2sql_limited_result LIMIT {max_rows}"


def _validation_provenance(sql: str, schema_fingerprint: str, max_rows: int) -> dict[str, Any]:
    return {
        "sql_sha256": hashlib.sha256(sql.strip().encode("utf-8")).hexdigest(),
        "schema_fingerprint": schema_fingerprint,
        "max_rows": max_rows,
    }


def _validation_matches(
    validation: dict[str, Any],
    sql: str,
    schema_fingerprint: str,
    max_rows: int,
) -> bool:
    provenance = validation.get("execution_provenance") or validation.get("provenance") or {}
    expected = _validation_provenance(sql, schema_fingerprint, max_rows)
    return (
        provenance.get("sql_sha256") == expected["sql_sha256"]
        and provenance.get("schema_fingerprint") == expected["schema_fingerprint"]
        and int(provenance.get("max_rows", -1)) == max_rows
    )


def _schema_fingerprint(schema: Any) -> str:
    if hasattr(schema, "describe"):
        payload = schema.describe()
    elif hasattr(schema, "model_dump"):
        payload = schema.model_dump()
    else:
        payload = schema
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sqlite_driver_connection(conn: Any) -> Any:
    raw = getattr(conn, "connection", None)
    if raw is None:
        return None
    return getattr(raw, "driver_connection", raw)
