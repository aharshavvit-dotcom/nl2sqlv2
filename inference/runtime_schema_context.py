from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any

from db.schema_graph import ForeignKeyInfo, SchemaGraph


SENSITIVE_MARKERS = ["email", "phone", "password", "token", "secret", "ssn", "address"]
DATE_MARKERS = ["date", "time", "created", "updated", "month", "year", "timestamp"]
ID_MARKERS = ["id", "_key", "key"]
NUMERIC_MARKERS = ["int", "real", "float", "double", "numeric", "decimal", "number"]
TEXT_MARKERS = ["char", "text", "string", "varchar"]


class RuntimeSchemaContext:
    def __init__(self, schema: SchemaGraph | dict[str, Any]):
        self.schema = schema
        self.dialect = self._detect_dialect(schema)
        self.tables: dict[str, dict[str, Any]] = self._normalize_schema(schema)
        self.foreign_keys: list[dict[str, str]] = self._extract_foreign_keys(schema)
        self.relationships = self._build_relationships()

    def get_tables(self) -> list[str]:
        return sorted(self.tables)

    def get_columns(self) -> list[str]:
        return [f"{table}.{column}" for table in self.get_tables() for column in self.tables[table]["columns"]]

    def get_numeric_columns(self) -> list[str]:
        return self._columns_by_predicate(lambda col: col["is_numeric"])

    def get_text_columns(self) -> list[str]:
        return self._columns_by_predicate(lambda col: col["is_text"])

    def get_date_columns(self) -> list[str]:
        return self._columns_by_predicate(lambda col: col["is_date"])

    def get_sensitive_columns(self) -> list[str]:
        return self._columns_by_predicate(lambda col: col["is_sensitive"])

    def get_table_columns(self, table: str) -> list[str]:
        return sorted(self.tables.get(table, {}).get("columns", {}))

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return table in self.tables and column in self.tables[table]["columns"]

    def column_info(self, table: str, column: str) -> dict[str, Any]:
        return self.tables[table]["columns"][column]

    def serialize_for_debug(self) -> dict[str, Any]:
        safe_tables = {
            table: {
                **info,
                "columns": {
                    column: {key: value for key, value in column_info.items() if key != "sample_values"}
                    for column, column_info in info.get("columns", {}).items()
                },
            }
            for table, info in self.tables.items()
        }
        return {
            "tables": safe_tables,
            "foreign_keys": self.foreign_keys,
            "relationships": dict(self.relationships),
            "dialect": self.dialect,
            "numeric_columns": self.get_numeric_columns(),
            "text_columns": self.get_text_columns(),
            "date_columns": self.get_date_columns(),
            "sensitive_columns": self.get_sensitive_columns(),
        }

    def _columns_by_predicate(self, predicate: Any) -> list[str]:
        values = []
        for table in self.get_tables():
            for column, info in self.tables[table]["columns"].items():
                if predicate(info):
                    values.append(f"{table}.{column}")
        return values

    @staticmethod
    def _detect_dialect(schema: SchemaGraph | dict[str, Any]) -> str:
        if isinstance(schema, SchemaGraph):
            return getattr(schema, "dialect", "sqlite") or "sqlite"
        return str(schema.get("dialect") or "sqlite").lower()

    @staticmethod
    def _normalize_schema(schema: SchemaGraph | dict[str, Any]) -> dict[str, dict[str, Any]]:
        if isinstance(schema, SchemaGraph):
            normalized: dict[str, dict[str, Any]] = {}
            for table_name, table in schema.tables.items():
                normalized[table_name] = {"columns": {}}
                for column_name, column in table.columns.items():
                    normalized[table_name]["columns"][column_name] = RuntimeSchemaContext._column_flags(
                        column_name,
                        str(column.type),
                        bool(column.primary_key),
                    )
            return normalized

        tables = schema.get("tables", schema)
        normalized = {}
        for table_name, table_info in tables.items():
            table_dict = table_info if isinstance(table_info, dict) else {}
            columns = table_dict.get("columns", table_dict)
            primary_keys = set(table_dict.get("primary_keys", []))
            normalized[table_name] = {"columns": {}}
            if isinstance(columns, dict):
                iterable = columns.items()
            else:
                iterable = [(str(item.get("name", item)), item) for item in columns]
            for column_name, raw in iterable:
                raw_dict = asdict(raw) if is_dataclass(raw) else (raw if isinstance(raw, dict) else {})
                normalized[table_name]["columns"][column_name] = RuntimeSchemaContext._column_flags(
                    column_name,
                    str(raw_dict.get("type", "")),
                    bool(
                        raw_dict.get("primary_key", False)
                        or raw_dict.get("is_primary_key", False)
                        or column_name in primary_keys
                    ),
                    sample_values=(
                        raw_dict.get("sample_values")
                        or raw_dict.get("values")
                        or raw_dict.get("examples")
                        or []
                    ),
                )
        return normalized

    @staticmethod
    def _column_flags(
        column_name: str,
        column_type: str,
        primary_key: bool,
        sample_values: list[Any] | None = None,
    ) -> dict[str, Any]:
        name = column_name.lower()
        typ = column_type.lower()
        return {
            "name": column_name,
            "type": column_type,
            "primary_key": primary_key,
            "is_primary_key": primary_key,
            "is_numeric": any(marker in typ for marker in NUMERIC_MARKERS) or primary_key,
            "is_text": any(marker in typ for marker in TEXT_MARKERS),
            "is_date": any(marker in name for marker in DATE_MARKERS) or "date" in typ or "time" in typ,
            "is_id": name == "id" or any(marker in name for marker in ID_MARKERS),
            "is_sensitive": any(marker in name for marker in SENSITIVE_MARKERS),
            "sample_values": [str(value) for value in (sample_values or [])[:50] if value is not None],
        }

    @staticmethod
    def _extract_foreign_keys(schema: SchemaGraph | dict[str, Any]) -> list[dict[str, str]]:
        if isinstance(schema, SchemaGraph):
            fks = []
            for table in schema.tables.values():
                for fk in table.foreign_keys:
                    fks.append(
                        {
                            "from_table": fk.table,
                            "from_column": fk.constrained_column,
                            "to_table": fk.referred_table,
                            "to_column": fk.referred_column,
                        }
                    )
            return fks

        fks: list[dict[str, str]] = []
        for fk in schema.get("foreign_keys", []) or []:
            normalized = RuntimeSchemaContext._normalize_fk(fk, fk.get("from_table") or fk.get("constrained_table"))
            if normalized:
                fks.append(normalized)
        for relationship in schema.get("relationships", []) or []:
            normalized = RuntimeSchemaContext._normalize_fk(relationship, relationship.get("from_table"))
            if normalized:
                fks.append(normalized)
        for table_name, table_info in (schema.get("tables", {}) or {}).items():
            if not isinstance(table_info, dict):
                continue
            for fk in table_info.get("foreign_keys", []) or []:
                normalized = RuntimeSchemaContext._normalize_fk(fk, table_name)
                if normalized:
                    fks.append(normalized)
        deduped: list[dict[str, str]] = []
        seen = set()
        for fk in fks:
            key = (fk["from_table"], fk["from_column"], fk["to_table"], fk["to_column"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fk)
        return deduped

    @staticmethod
    def _normalize_fk(raw: dict[str, Any], default_from_table: str | None = None) -> dict[str, str] | None:
        from_table = raw.get("from_table") or raw.get("constrained_table") or default_from_table
        from_column = raw.get("from_column") or raw.get("constrained_column") or raw.get("column")
        to_table = raw.get("to_table") or raw.get("referred_table") or raw.get("references_table")
        to_column = raw.get("to_column") or raw.get("referred_column") or raw.get("references_column")
        if not all([from_table, from_column, to_table, to_column]):
            return None
        return {
            "from_table": str(from_table),
            "from_column": str(from_column),
            "to_table": str(to_table),
            "to_column": str(to_column),
        }

    def _build_relationships(self) -> dict[str, list[dict[str, str]]]:
        graph: dict[str, list[dict[str, str]]] = defaultdict(list)
        for fk in self.foreign_keys:
            left = fk["from_table"]
            right = fk["to_table"]
            graph[left].append({**fk, "neighbor": right})
            graph[right].append({**fk, "neighbor": left})

        for left in self.tables:
            for right in self.tables:
                if left == right:
                    continue
                for left_col in self.get_table_columns(left):
                    for right_col in self.get_table_columns(right):
                        if left_col == right_col and left_col.endswith("_id"):
                            inferred = {
                                "from_table": left,
                                "from_column": left_col,
                                "to_table": right,
                                "to_column": right_col,
                                "neighbor": right,
                                "inferred": "true",
                            }
                            graph[left].append(inferred)
        return graph
