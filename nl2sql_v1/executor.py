from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from .schema import read_sqlite_schema
from validation.sql_validator import SQLValidator


def sqlite_url(db_path: str | Path) -> str:
    path = Path(db_path).resolve()
    return f"sqlite:///{path.as_posix()}"


def execute_select(db_path: str | Path, sql: str, max_rows: int = 1000) -> pd.DataFrame:
    schema = read_sqlite_schema(db_path)
    validation = SQLValidator().validate(sql, schema=schema, max_limit=max_rows)
    if not validation["is_valid"]:
        raise ValueError("; ".join(validation["issues"]))

    engine = create_engine(sqlite_url(db_path), future=True)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")
        df = pd.read_sql_query(text(sql), conn)
    if len(df) > max_rows:
        return df.head(max_rows)
    return df
