from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any

from nl2sql_v1.schema import SchemaGraph


DATE_NAME_MARKERS = ("date", "time", "created", "updated", "month", "year", "timestamp")
NUMERIC_NAME_MARKERS = ("amount", "revenue", "sales", "price", "quantity", "total", "cost", "fare", "count")
NUMERIC_TYPE_MARKERS = ("int", "real", "float", "double", "numeric", "decimal", "number")
TEXT_MARKERS = ("char", "text", "string", "varchar")
TEXT_NAME_MARKERS = ("name", "status", "region", "category", "city", "country", "type", "description", "product", "customer")


class SchemaLinearizer:
    def linearize(self, schema: Any) -> str:
        items = extract_schema_items(schema)
        parts = []
        for table in items["tables"]:
            cols = [col["column"] for col in items["columns"] if col["table"] == table]
            parts.append(f"{table}({', '.join(cols)})")
        return "tables: " + "; ".join(parts)

    def extract_schema_items(self, schema: Any) -> dict[str, Any]:
        return extract_schema_items(schema)


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
            return _from_tables_dict(schema_context.get("tables") or {}, schema_context.get("foreign_keys") or [])
        if schema.get("foreign_keys") is not None and schema.get("tables"):
            return _from_tables_dict(schema.get("tables") or {}, schema.get("foreign_keys") or [])
        if schema.get("tables"):
            tables_value = schema.get("tables")
            if isinstance(tables_value, dict):
                return _from_tables_dict(tables_value, schema.get("foreign_keys") or [])
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
        fk_columns = {fk.constrained_column for fk in table.foreign_keys}
        fk_targets = {
            fk.constrained_column: {"table": fk.referred_table, "column": fk.referred_column}
            for fk in table.foreign_keys
        }
        for column_name, column in sorted(table.columns.items()):
            columns.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "type": _column_type(column_name, str(column.type)),
                    "primary_key": bool(column.primary_key),
                    "foreign_key": column_name in fk_columns,
                    "foreign_key_target": fk_targets.get(column_name),
                }
            )
    return _finalize(sorted(schema.tables), columns)


def _from_tables_dict(tables: dict[str, Any], foreign_keys: list[Any] | None = None) -> dict[str, Any]:
    table_names = [str(name) for name in tables]
    columns = []
    fk_map = _foreign_key_map(foreign_keys or [])
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
            resolved_column = str(_column_name(raw_dict) or column_name)
            fk_target = (
                raw_dict.get("foreign_key_target")
                or raw_dict.get("references")
                or fk_map.get((str(table_name), resolved_column))
            )
            columns.append(
                {
                    "table": str(table_name),
                    "column": resolved_column,
                    "type": _column_type(resolved_column, str(raw_dict.get("type", ""))),
                    "primary_key": bool(raw_dict.get("primary_key", raw_dict.get("pk", False))),
                    "foreign_key": bool(raw_dict.get("foreign_key", raw_dict.get("is_fk", False)) or fk_target),
                    "foreign_key_target": fk_target,
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
                    "primary_key": bool(column_dict.get("primary_key", column_dict.get("pk", False))),
                    "foreign_key": bool(column_dict.get("foreign_key", column_dict.get("is_fk", False))),
                    "foreign_key_target": column_dict.get("foreign_key_target") or column_dict.get("references"),
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


def _foreign_key_map(foreign_keys: list[Any]) -> dict[tuple[str, str], dict[str, str]]:
    mapped: dict[tuple[str, str], dict[str, str]] = {}
    for raw in foreign_keys:
        item = asdict(raw) if is_dataclass(raw) else (raw if isinstance(raw, dict) else {})
        from_table = str(item.get("from_table") or item.get("table") or "")
        from_column = str(item.get("from_column") or item.get("constrained_column") or "")
        to_table = str(item.get("to_table") or item.get("referred_table") or "")
        to_column = str(item.get("to_column") or item.get("referred_column") or "")
        if from_table and from_column and to_table and to_column:
            mapped[(from_table, from_column)] = {"table": to_table, "column": to_column}
    return mapped


def _finalize(tables: list[str], columns: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_tables = list(dict.fromkeys(str(table) for table in tables if table))
    normalized_columns = []
    for item in columns:
        if not item.get("table") or not item.get("column"):
            continue
        index = len(normalized_columns)
        normalized_columns.append(
            {
                "index": index,
                "table": str(item["table"]),
                "column": str(item["column"]),
                "type": str(item.get("type") or _column_type(str(item["column"]), "")),
                "primary_key": bool(item.get("primary_key", False)),
                "foreign_key": bool(item.get("foreign_key", False)),
                "foreign_key_target": item.get("foreign_key_target"),
            }
        )
    return {
        "tables": normalized_tables,
        "columns": normalized_columns,
        "date_columns": [item for item in normalized_columns if item["type"] == "date"],
        "numeric_columns": [item for item in normalized_columns if item["type"] == "numeric"],
        "text_columns": [item for item in normalized_columns if item["type"] == "text"],
        "id_columns": [item for item in normalized_columns if item["type"] == "id"],
    }


def _column_name(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("name") or raw.get("column") or "")
    return str(getattr(raw, "name", raw))


def _column_type(column_name: str, column_type: str) -> str:
    name = column_name.lower()
    typ = column_type.lower()
    name_parts = set(part for part in re.split(r"[^a-z0-9]+", name) if part)
    if typ == "id" or name == "id" or name.endswith("_id"):
        return "id"
    if (
        name in {"order_date", "created_at", "updated_at", "transaction_date"}
        or name.endswith("_date")
        or name.endswith("_time")
        or name.endswith("_at")
        or bool(name_parts & set(DATE_NAME_MARKERS))
        or "date" in typ
        or "time" in typ
    ):
        return "date"
    if any(marker in typ for marker in NUMERIC_TYPE_MARKERS):
        return "numeric"
    if name_parts & set(TEXT_NAME_MARKERS):
        return "text"
    if name in NUMERIC_NAME_MARKERS or bool(name_parts & set(NUMERIC_NAME_MARKERS)):
        return "numeric"
    if any(marker in typ for marker in TEXT_MARKERS):
        return "text"
    return "text"
