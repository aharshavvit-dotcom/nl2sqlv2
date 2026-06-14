from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any

from nl2sql_v1.schema import SchemaGraph


DATE_MARKERS = ("date", "time", "created", "updated", "month", "year", "timestamp")
NUMERIC_MARKERS = ("int", "real", "float", "double", "numeric", "decimal", "number")
TEXT_MARKERS = ("char", "text", "string", "varchar")


class SchemaLinearizer:
    def linearize(self, schema: Any) -> str:
        items = extract_schema_items(schema)
        parts = []
        for table in items["tables"]:
            cols = [col["column"] for col in items["columns"] if col["table"] == table]
            parts.append(f"{table}({', '.join(cols)})")
        return "tables: " + "; ".join(parts)


def extract_schema_items(schema: Any) -> dict[str, Any]:
    if isinstance(schema, str):
        return _from_serialized_schema(schema)
    if isinstance(schema, SchemaGraph):
        return _from_schema_graph(schema)
    if schema is None:
        return _finalize([], [])
    if isinstance(schema, dict):
        if "serialized_schema" in schema and not schema.get("tables"):
            return _from_serialized_schema(str(schema.get("serialized_schema") or ""))
        schema_context = (
            (schema.get("metadata") or {})
            .get("validation_context", {})
            .get("schema_context", {})
        )
        if schema_context.get("tables"):
            return _from_tables_dict(schema_context.get("tables") or {})
        if schema.get("foreign_keys") is not None and schema.get("tables"):
            return _from_tables_dict(schema.get("tables") or {})
        if schema.get("tables"):
            tables_value = schema.get("tables")
            if isinstance(tables_value, dict):
                return _from_tables_dict(tables_value)
            if isinstance(tables_value, list):
                return _from_table_list(tables_value)
        return _from_tables_dict(schema)
    return _finalize([], [])


def schema_from_example(row: dict[str, Any]) -> dict[str, Any]:
    query_ir = row.get("query_ir") or {}
    schema_context = (
        (query_ir.get("metadata") or {})
        .get("validation_context", {})
        .get("schema_context", {})
    )
    if schema_context.get("tables"):
        return {
            "tables": schema_context["tables"],
            "foreign_keys": schema_context.get("foreign_keys", []),
            "dialect": schema_context.get("dialect", "sqlite"),
        }
    serialized = row.get("serialized_schema") or (query_ir.get("metadata") or {}).get("serialized_schema")
    if serialized:
        return _schema_dict_from_serialized(str(serialized))
    return {"tables": {}}


def _from_schema_graph(schema: SchemaGraph) -> dict[str, Any]:
    columns = []
    for table_name, table in sorted(schema.tables.items()):
        for column_name, column in sorted(table.columns.items()):
            columns.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "type": _column_type(column_name, str(column.type)),
                }
            )
    return _finalize(sorted(schema.tables), columns)


def _from_tables_dict(tables: dict[str, Any]) -> dict[str, Any]:
    table_names = [str(name) for name in tables]
    columns = []
    for table_name in table_names:
        table_info = tables.get(table_name, {})
        if is_dataclass(table_info):
            table_info = asdict(table_info)
        raw_columns = table_info.get("columns", table_info if isinstance(table_info, dict) else {})
        if isinstance(raw_columns, dict):
            iterable = raw_columns.items()
        else:
            iterable = [(str(_column_name(item)), item) for item in raw_columns or []]
        for column_name, raw in iterable:
            raw_dict = asdict(raw) if is_dataclass(raw) else (raw if isinstance(raw, dict) else {})
            columns.append(
                {
                    "table": str(table_name),
                    "column": str(_column_name(raw_dict) or column_name),
                    "type": _column_type(str(_column_name(raw_dict) or column_name), str(raw_dict.get("type", ""))),
                }
            )
    return _finalize(table_names, columns)


def _from_table_list(tables: list[Any]) -> dict[str, Any]:
    table_names = []
    columns = []
    for table in tables:
        table_dict = asdict(table) if is_dataclass(table) else (table if isinstance(table, dict) else {})
        table_name = str(table_dict.get("name") or table_dict.get("table") or "")
        if not table_name:
            continue
        table_names.append(table_name)
        for column in table_dict.get("columns", []):
            column_dict = asdict(column) if is_dataclass(column) else (column if isinstance(column, dict) else {"name": column})
            column_name = str(column_dict.get("name") or column_dict.get("column") or column)
            columns.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "type": _column_type(column_name, str(column_dict.get("type", ""))),
                }
            )
    return _finalize(table_names, columns)


def _from_serialized_schema(text: str) -> dict[str, Any]:
    schema = _schema_dict_from_serialized(text)
    return _from_tables_dict(schema["tables"])


def _schema_dict_from_serialized(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.lower().startswith("tables:"):
        body = body.split(":", 1)[1]
    tables: dict[str, Any] = {}
    for match in re.finditer(r"([A-Za-z_][\w]*)\s*\(([^)]*)\)", body):
        table = match.group(1)
        columns = {}
        for raw_column in match.group(2).split(","):
            column = raw_column.strip()
            if not column:
                continue
            columns[column] = {"type": _column_type(column, "")}
        tables[table] = {"columns": columns}
    return {"tables": tables, "dialect": "sqlite"}


def _finalize(tables: list[str], columns: list[dict[str, str]]) -> dict[str, Any]:
    normalized_tables = list(dict.fromkeys(str(table) for table in tables if table))
    normalized_columns = []
    for item in columns:
        if not item.get("table") or not item.get("column"):
            continue
        normalized_columns.append(
            {
                "table": str(item["table"]),
                "column": str(item["column"]),
                "type": str(item.get("type") or _column_type(str(item["column"]), "")),
            }
        )
    return {
        "tables": normalized_tables,
        "columns": normalized_columns,
        "date_columns": [item for item in normalized_columns if item["type"] == "date"],
        "numeric_columns": [item for item in normalized_columns if item["type"] == "numeric"],
        "text_columns": [item for item in normalized_columns if item["type"] == "text"],
    }


def _column_name(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("name") or raw.get("column") or "")
    return str(getattr(raw, "name", raw))


def _column_type(column_name: str, column_type: str) -> str:
    name = column_name.lower()
    typ = column_type.lower()
    if any(marker in name for marker in DATE_MARKERS) or "date" in typ or "time" in typ:
        return "date"
    if any(marker in typ for marker in NUMERIC_MARKERS):
        return "numeric"
    if any(marker in typ for marker in TEXT_MARKERS):
        return "text"
    if name == "id" or name.endswith("_id") or name.lower().endswith("id"):
        return "numeric"
    return "text"
