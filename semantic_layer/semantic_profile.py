from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SemanticColumn:
    table: str
    column: str
    semantic_role: str
    aliases: list[str]
    confidence: float
    is_sensitive: bool = False


@dataclass
class SemanticTable:
    table: str
    table_type: str
    aliases: list[str]
    safe_columns: list[str]
    columns: list[SemanticColumn] = field(default_factory=list)


@dataclass
class SemanticProfile:
    dialect: str
    database: str | None
    schema_name: str | None
    tables: dict[str, SemanticTable]
    metrics: dict[str, Any]
    dimensions: dict[str, Any]
    dates: dict[str, Any]
    filters: dict[str, Any]
    relationships: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
