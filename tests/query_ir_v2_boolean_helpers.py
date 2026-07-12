from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ir.query_ir_v2_models import BooleanPredicate, ColumnExpression, ComparisonPredicate, LiteralExpression


ROOT = Path(__file__).resolve().parents[1]
BOOLEAN_EVAL_PATH = ROOT / "evaluation" / "query_ir_v2_boolean_eval_cases.jsonl"


def col(name: str, table: str = "customers") -> ColumnExpression:
    return ColumnExpression(table=table, column=name)


def lit(value: Any, value_type: str = "string") -> LiteralExpression:
    return LiteralExpression(value=value, value_type=value_type)


def eq(column: str, value: Any, table: str = "customers", value_type: str = "string") -> ComparisonPredicate:
    return ComparisonPredicate(left=col(column, table), operator="=", right=lit(value, value_type))


def or_tree(*operands: Any) -> BooleanPredicate:
    return BooleanPredicate(operator="OR", operands=list(operands))


def and_tree(*operands: Any) -> BooleanPredicate:
    return BooleanPredicate(operator="AND", operands=list(operands))


def load_boolean_eval_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in BOOLEAN_EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            region TEXT,
            status TEXT,
            tier TEXT,
            deleted_at TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            status TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO customers(id, region, status, tier, deleted_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "US", "ACTIVE", "GOLD", None, "2025-12-15"),
            (2, "CA", "INACTIVE", "SILVER", "2026-01-10", "2026-02-01"),
            (3, "MX", "ACTIVE", "BRONZE", None, "2026-07-01"),
            (4, "US", "INACTIVE", "GOLD", "2026-03-01", "2026-05-15"),
            (5, "FR", "ACTIVE", "GOLD", None, "2026-04-20"),
        ],
    )
    conn.executemany(
        "INSERT INTO orders(id, customer_id, status) VALUES (?, ?, ?)",
        [
            (1, 1, "OPEN"),
            (2, 2, "CLOSED"),
            (3, 3, "OPEN"),
            (4, 5, "CLOSED"),
        ],
    )
    return conn


def execute_rows(conn: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    return sorted(conn.execute(sql).fetchall())
