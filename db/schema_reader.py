"""Unified schema reader producing a normalized dict and SchemaGraph objects."""

from __future__ import annotations

from typing import Any

from nl2sql_v1.schema import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo

from .connection_config import DatabaseConnectionConfig
from .sqlite_connector import SQLiteConnector


def read_database_schema(config: DatabaseConnectionConfig) -> dict[str, Any]:
    """Read database schema using the appropriate connector.

    Returns a normalized dict with the shape::

        {
            "dialect": "sqlite" | "postgres",
            "database": "...",
            "schema_name": "public" | None,
            "tables": { "<table>": { "columns": [...], "primary_keys": [...], "foreign_keys": [...] } },
            "relationships": [...]
        }
    """
    connector = _connector_for(config)
    return connector.read_schema()


def schema_dict_to_graph(schema_dict: dict[str, Any]) -> SchemaGraph:
    """Convert the normalized schema dict into a ``SchemaGraph`` for pipeline compatibility."""
    dialect = schema_dict.get("dialect", "sqlite")
    tables: dict[str, TableInfo] = {}

    for table_name, table_data in schema_dict.get("tables", {}).items():
        columns: dict[str, ColumnInfo] = {}
        for col in table_data.get("columns", []):
            columns[col["name"]] = ColumnInfo(
                name=col["name"],
                type=col.get("type", "TEXT"),
                nullable=col.get("nullable", True),
                primary_key=col.get("is_primary_key", False),
            )

        foreign_keys: list[ForeignKeyInfo] = []
        for fk in table_data.get("foreign_keys", []):
            foreign_keys.append(ForeignKeyInfo(
                table=fk.get("constrained_table", table_name),
                constrained_column=fk["constrained_column"],
                referred_table=fk["referred_table"],
                referred_column=fk["referred_column"],
            ))

        tables[table_name] = TableInfo(
            name=table_name,
            columns=columns,
            foreign_keys=foreign_keys,
        )

    return SchemaGraph(tables=tables, dialect=dialect)


def schema_summary(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a concise summary of the schema for UI display."""
    tables = schema_dict.get("tables", {})
    total_columns = sum(len(t.get("columns", [])) for t in tables.values())
    return {
        "dialect": schema_dict.get("dialect"),
        "database": schema_dict.get("database"),
        "schema_name": schema_dict.get("schema_name"),
        "table_count": len(tables),
        "column_count": total_columns,
        "relationship_count": len(schema_dict.get("relationships", [])),
        "tables": {
            name: {
                "column_count": len(info.get("columns", [])),
                "primary_keys": info.get("primary_keys", []),
                "foreign_key_count": len(info.get("foreign_keys", [])),
            }
            for name, info in tables.items()
        },
    }


def _connector_for(config: DatabaseConnectionConfig):
    """Return the right connector instance for the given config."""
    if config.db_type == "sqlite":
        return SQLiteConnector(config)
    if config.db_type == "postgres":
        from .postgres_connector import PostgresConnector
        return PostgresConnector(config)
    raise ValueError(f"Unsupported db_type: {config.db_type!r}")
