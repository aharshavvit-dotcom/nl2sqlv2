from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .schema import SchemaGraph


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str = "ok"


DISALLOWED = (
    exp.Alter,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Insert,
    exp.Merge,
    exp.Update,
)


def validate_select_sql(sql: str, schema: SchemaGraph | None = None) -> ValidationResult:
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:
        return ValidationResult(False, f"SQL parse failed: {exc}")

    if len(parsed) != 1:
        return ValidationResult(False, "Only one SQL statement is allowed")

    statement = parsed[0]
    if not isinstance(statement, exp.Select):
        return ValidationResult(False, "Only SELECT statements are allowed")
    if any(statement.find(disallowed) for disallowed in DISALLOWED):
        return ValidationResult(False, "Mutating SQL is not allowed")

    if schema is not None:
        missing = []
        for table in statement.find_all(exp.Table):
            table_name = table.name
            if table_name and not schema.has_table(table_name):
                missing.append(f"table {table_name}")
        if missing:
            return ValidationResult(False, "Unknown schema references: " + ", ".join(sorted(set(missing))))

    return ValidationResult(True)
