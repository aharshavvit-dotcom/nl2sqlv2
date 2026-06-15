"""SQLite database connector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from .connection_config import DatabaseConnectionConfig
from .connector_base import DatabaseConnector


class SQLiteConnector(DatabaseConnector):
    """Connector for SQLite databases."""

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        if config.db_type != "sqlite":
            raise ValueError("SQLiteConnector requires db_type='sqlite'")
        self.config = config
        self.db_path = Path(config.sqlite_path or "").expanduser().resolve()

    def test_connection(self) -> tuple[bool, str]:
        if not self.db_path.exists():
            return False, f"Database file not found: {self.db_path}"
        try:
            engine = create_engine(self.config.sqlalchemy_url(), future=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connected to SQLite database."
        except Exception as exc:
            return False, f"SQLite connection failed: {exc}"

    def read_schema(self) -> dict[str, Any]:
        engine = create_engine(self.config.sqlalchemy_url(), future=True)
        inspector = inspect(engine)
        tables: dict[str, Any] = {}
        relationships: list[dict[str, Any]] = []

        for table_name in inspector.get_table_names():
            columns: list[dict[str, Any]] = []
            pk_columns: list[str] = []
            fk_list: list[dict[str, Any]] = []

            for col in inspector.get_columns(table_name):
                is_pk = bool(col.get("primary_key", False))
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": bool(col.get("nullable", True)),
                    "is_primary_key": is_pk,
                    "is_foreign_key": False,  # updated below
                })
                if is_pk:
                    pk_columns.append(col["name"])

            for fk in inspector.get_foreign_keys(table_name):
                referred_table = fk.get("referred_table")
                constrained = fk.get("constrained_columns") or []
                referred = fk.get("referred_columns") or []
                if referred_table and constrained and referred:
                    fk_entry = {
                        "constrained_table": table_name,
                        "constrained_column": constrained[0],
                        "referred_table": referred_table,
                        "referred_column": referred[0],
                    }
                    fk_list.append(fk_entry)
                    relationships.append(fk_entry)
                    # Mark column as FK
                    for c in columns:
                        if c["name"] == constrained[0]:
                            c["is_foreign_key"] = True

            tables[table_name] = {
                "columns": columns,
                "primary_keys": pk_columns,
                "foreign_keys": fk_list,
            }

        return {
            "dialect": "sqlite",
            "database": self.db_path.name,
            "schema_name": None,
            "tables": tables,
            "relationships": relationships,
        }

    def execute_readonly(self, sql: str, limit: int | None = None) -> dict[str, Any]:
        from validation.sql_validator import SQLValidator

        validator = SQLValidator()
        max_limit = limit or 1000
        validation = validator.validate(sql, max_limit=max_limit, dialect="sqlite")
        if not validation.get("is_valid", validation.get("ok", False)):
            issues = validation.get("issues") or [validation.get("message", "SQL validation failed")]
            return {"error": "; ".join(str(i) for i in issues), "columns": [], "rows": []}

        engine = create_engine(self.config.sqlalchemy_url(), future=True)
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA query_only = ON")
                df = pd.read_sql_query(text(sql), conn)
            if limit and len(df) > limit:
                df = df.head(limit)
            return {
                "columns": list(df.columns),
                "rows": df.values.tolist(),
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "rows": []}

    def get_dialect(self) -> str:
        return "sqlite"
