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
