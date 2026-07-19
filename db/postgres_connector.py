"""PostgreSQL database connector."""

from __future__ import annotations

from typing import Any
import os
import time

import pandas as pd
from sqlalchemy import create_engine, text

from .connection_config import DatabaseConnectionConfig
from .connector_base import DatabaseConnector


class PostgresConnector(DatabaseConnector):
    """Connector for PostgreSQL databases.

    Enforces read-only transactions, statement timeouts, and
    SELECT-only execution for safety.
    """

    STATEMENT_TIMEOUT = "30s"

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        if config.db_type != "postgres":
            raise ValueError("PostgresConnector requires db_type='postgres'")
        self.config = config
        self.schema_name = config.schema_name or "public"
        try:
            seconds = max(1, int(os.getenv("NL2SQL_DB_STATEMENT_TIMEOUT_SECONDS", "30")))
        except ValueError:
            seconds = 30
        self.statement_timeout_ms = seconds * 1000

    def test_connection(self) -> tuple[bool, str]:
        try:
            engine = create_engine(
                self.config.sqlalchemy_url(),
                future=True,
                pool_pre_ping=True,
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return True, f"Connected to PostgreSQL database '{self.config.database}'."
        except Exception as exc:
            # Sanitize error – never leak password
            msg = str(exc)
            if self.config.password and self.config.password in msg:
                msg = msg.replace(self.config.password, "***")
            return False, f"PostgreSQL connection failed: {msg}"

    def read_schema(self) -> dict[str, Any]:
        engine = create_engine(self.config.sqlalchemy_url(), future=True)
        tables: dict[str, Any] = {}
        relationships: list[dict[str, Any]] = []
        schema = self.schema_name

        with engine.connect() as conn:
            # --- Tables ---
            table_rows = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            ), {"schema": schema}).fetchall()

            table_names = [row[0] for row in table_rows]

            # --- Columns ---
            col_rows = conn.execute(text(
                "SELECT table_name, column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = :schema "
                "ORDER BY table_name, ordinal_position"
            ), {"schema": schema}).fetchall()

            col_map: dict[str, list[dict[str, Any]]] = {}
            for row in col_rows:
                tbl, col_name, dtype, nullable = row
                col_map.setdefault(tbl, []).append({
                    "name": col_name,
                    "type": dtype,
                    "nullable": nullable == "YES",
                    "is_primary_key": False,
                    "is_foreign_key": False,
                })

            # --- Primary keys ---
            pk_rows = conn.execute(text(
                "SELECT tc.table_name, kcu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                "  AND tc.table_schema = kcu.table_schema "
                "WHERE tc.constraint_type = 'PRIMARY KEY' "
                "  AND tc.table_schema = :schema"
            ), {"schema": schema}).fetchall()

            pk_map: dict[str, list[str]] = {}
            for row in pk_rows:
                tbl, col_name = row
                pk_map.setdefault(tbl, []).append(col_name)
                for c in col_map.get(tbl, []):
                    if c["name"] == col_name:
                        c["is_primary_key"] = True

            # --- Foreign keys ---
            fk_rows = conn.execute(text(
                "SELECT "
                "  kcu.table_name AS constrained_table, "
                "  kcu.column_name AS constrained_column, "
                "  ccu.table_name AS referred_table, "
                "  ccu.column_name AS referred_column "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                "  AND tc.table_schema = kcu.table_schema "
                "JOIN information_schema.constraint_column_usage ccu "
                "  ON tc.constraint_name = ccu.constraint_name "
                "  AND tc.table_schema = ccu.table_schema "
                "WHERE tc.constraint_type = 'FOREIGN KEY' "
                "  AND tc.table_schema = :schema"
            ), {"schema": schema}).fetchall()

            fk_map: dict[str, list[dict[str, Any]]] = {}
            for row in fk_rows:
                c_tbl, c_col, r_tbl, r_col = row
                fk_entry = {
                    "constrained_table": c_tbl,
                    "constrained_column": c_col,
                    "referred_table": r_tbl,
                    "referred_column": r_col,
                }
                fk_map.setdefault(c_tbl, []).append(fk_entry)
                relationships.append(fk_entry)
                for c in col_map.get(c_tbl, []):
                    if c["name"] == c_col:
                        c["is_foreign_key"] = True

            for tbl in table_names:
                tables[tbl] = {
                    "columns": col_map.get(tbl, []),
                    "primary_keys": pk_map.get(tbl, []),
                    "foreign_keys": fk_map.get(tbl, []),
                }

        engine.dispose()
        return {
            "dialect": "postgres",
            "database": self.config.database,
            "schema_name": schema,
            "tables": tables,
            "relationships": relationships,
        }

    def execute_readonly(self, sql: str, limit: int | None = None) -> dict[str, Any]:
        from validation.sql_validator import SQLValidator

        validator = SQLValidator()
        max_limit = limit or 1000
        validation = validator.validate(sql, max_limit=max_limit, dialect="postgres")
        if not validation.get("is_valid", validation.get("ok", False)):
            issues = validation.get("issues") or [validation.get("message", "SQL validation failed")]
            return {"error": "; ".join(str(i) for i in issues), "columns": [], "rows": []}

        engine = create_engine(self.config.sqlalchemy_url(), future=True)
        try:
            start = time.monotonic()
            with engine.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text(f"SET LOCAL statement_timeout = {self.statement_timeout_ms}"))
                df = pd.read_sql_query(text(_outer_limited_select(sql, max_limit)), conn)
                conn.rollback()  # ensure read-only transaction is closed cleanly
            return {
                "columns": list(df.columns),
                "rows": df.values.tolist(),
                "duration_ms": round((time.monotonic() - start) * 1000, 3),
                "row_count": int(len(df)),
                "max_rows": max_limit,
            }
        except Exception as exc:
            msg = str(exc)
            if self.config.password and self.config.password in msg:
                msg = msg.replace(self.config.password, "***")
            return {"error": msg, "columns": [], "rows": []}
        finally:
            engine.dispose()

    def get_dialect(self) -> str:
        return "postgres"


def _outer_limited_select(sql: str, max_rows: int) -> str:
    stripped = sql.strip().rstrip(";")
    return f"SELECT * FROM ({stripped}) AS nl2sql_limited_result LIMIT {max(1, int(max_rows or 1))}"
