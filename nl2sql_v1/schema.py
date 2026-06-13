from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, inspect


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool


@dataclass(frozen=True)
class ForeignKeyInfo:
    table: str
    constrained_column: str
    referred_table: str
    referred_column: str


@dataclass
class TableInfo:
    name: str
    columns: dict[str, ColumnInfo] = field(default_factory=dict)
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)


@dataclass
class SchemaGraph:
    tables: dict[str, TableInfo]
    dialect: str = "sqlite"

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return table in self.tables and column in self.tables[table].columns

    def neighbors(self, table: str) -> list[tuple[ForeignKeyInfo, str]]:
        neighbors: list[tuple[ForeignKeyInfo, str]] = []
        for candidate in self.tables.values():
            for fk in candidate.foreign_keys:
                if fk.table == table:
                    neighbors.append((fk, fk.referred_table))
                elif fk.referred_table == table:
                    neighbors.append((fk, fk.table))
        return neighbors

    def describe(self) -> dict[str, list[str]]:
        return {name: sorted(table.columns) for name, table in sorted(self.tables.items())}


def sqlite_url(db_path: str | Path) -> str:
    path = Path(db_path).resolve()
    return f"sqlite:///{path.as_posix()}"


def read_sqlite_schema(db_path: str | Path) -> SchemaGraph:
    engine = create_engine(sqlite_url(db_path), future=True)
    inspector = inspect(engine)
    tables: dict[str, TableInfo] = {}

    for table_name in inspector.get_table_names():
        columns: dict[str, ColumnInfo] = {}
        for column in inspector.get_columns(table_name):
            columns[column["name"]] = ColumnInfo(
                name=column["name"],
                type=str(column["type"]),
                nullable=bool(column.get("nullable", True)),
                primary_key=bool(column.get("primary_key", False)),
            )
        table = TableInfo(name=table_name, columns=columns)
        for fk in inspector.get_foreign_keys(table_name):
            referred_table = fk.get("referred_table")
            constrained = fk.get("constrained_columns") or []
            referred = fk.get("referred_columns") or []
            if referred_table and constrained and referred:
                table.foreign_keys.append(
                    ForeignKeyInfo(
                        table=table_name,
                        constrained_column=constrained[0],
                        referred_table=referred_table,
                        referred_column=referred[0],
                    )
                )
        tables[table_name] = table
    return SchemaGraph(tables=tables, dialect="sqlite")
