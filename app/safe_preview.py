from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from nl2sql_v1.schema import SchemaGraph


SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address")


def build_safe_preview_sql(
    table_name: str,
    schema: dict[str, Any] | SchemaGraph,
    max_columns: int = 10,
    limit: int = 20,
) -> str | None:
    columns = [
        column
        for column in _table_columns(table_name, schema)
        if not _is_sensitive(column)
    ]
    if not columns:
        return None
    selected = ", ".join(_quote_identifier(column) for column in columns[:max_columns])
    return f"SELECT {selected}\nFROM {_quote_identifier(table_name)}\nLIMIT {max(1, int(limit))}"


def _table_columns(table_name: str, schema: dict[str, Any] | SchemaGraph) -> list[str]:
    if isinstance(schema, SchemaGraph):
        table = schema.tables.get(table_name)
        return list(table.columns) if table else []

    tables = schema.get("tables", schema)
    table = tables.get(table_name, {})
    columns = table.get("columns", table) if isinstance(table, dict) else getattr(table, "columns", {})
    if isinstance(columns, dict):
        return [str(column) for column in columns]
    values = []
    for column in columns:
        raw = asdict(column) if is_dataclass(column) else column
        values.append(str(raw.get("name", raw)) if isinstance(raw, dict) else str(raw))
    return values


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _is_sensitive(column_name: str) -> bool:
    lowered = column_name.lower()
    return any(marker in lowered for marker in SENSITIVE_MARKERS)
