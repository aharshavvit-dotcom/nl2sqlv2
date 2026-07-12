"""Rendered query output contracts.

Separates logical parameterized SQL from driver-specific binding.
The connector translates logical BoundParameter objects into driver placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ir.query_ir_v2_models import LiteralValueType


@dataclass(frozen=True)
class BoundParameter:
    """A logical parameter — connector translates to driver-specific binding."""
    name: str
    value_type: LiteralValueType
    value: str | int | bool | None


@dataclass
class RenderedQuery:
    """Output contract for rendered SQL.

    sql_template uses named placeholders (:param_name) that the connector
    translates to driver-specific syntax (? for SQLite, $1 for Postgres, etc.).
    """
    sql_template: str
    parameters: tuple[BoundParameter, ...] = field(default_factory=tuple)
    dialect: str = "sqlite"
    query_ir_fingerprint: str = ""

    @property
    def sql(self) -> str:
        """Convenience: render with inline literals (for display/logging only)."""
        result = self.sql_template
        for param in self.parameters:
            placeholder = f":{param.name}"
            if param.value is None:
                result = result.replace(placeholder, "NULL", 1)
            elif isinstance(param.value, str):
                escaped = param.value.replace("'", "''")
                result = result.replace(placeholder, f"'{escaped}'", 1)
            elif isinstance(param.value, bool):
                result = result.replace(placeholder, str(int(param.value)), 1)
            else:
                result = result.replace(placeholder, str(param.value), 1)
        return result
