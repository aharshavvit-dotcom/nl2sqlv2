from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .query_ir_migration import V1_TO_V2_WARNING, V2_TO_V1_WARNING, coerce_query_ir_v1, convert_v2_to_v1, migrate_v1_to_v2
from .query_ir_models import QueryIR
from .query_ir_v2_models import QUERY_IR_V2_VERSION, QueryNode


QueryIRVersion = Literal["1", "2.0"]


class QueryIRVersionError(ValueError):
    pass


class QueryIRLoadDiagnostics(BaseModel):
    detected_version: str
    target_version: str
    warnings: list[str] = Field(default_factory=list)
    migration_warnings: list[str] = Field(default_factory=list)


class LoadedQueryIR(BaseModel):
    query_ir_version: QueryIRVersion
    query_ir: Any
    diagnostics: QueryIRLoadDiagnostics


def detect_query_ir_version(value: QueryIR | QueryNode | dict[str, Any]) -> QueryIRVersion:
    if isinstance(value, QueryNode):
        return "2.0"
    if isinstance(value, QueryIR):
        return "1"
    if not isinstance(value, dict):
        raise TypeError(f"Expected QueryIR object or dict, got {type(value).__name__}")
    version = value.get("query_ir_version")
    if version is None:
        return "1"
    normalized = str(version)
    if normalized in {"1", "1.0"}:
        return "1"
    if normalized == QUERY_IR_V2_VERSION:
        return "2.0"
    raise QueryIRVersionError(f"Unknown QueryIR version: {version!r}")


def load_query_ir(
    value: QueryIR | QueryNode | dict[str, Any],
    *,
    target_version: QueryIRVersion = "1",
) -> LoadedQueryIR:
    detected = detect_query_ir_version(value)
    warnings: list[str] = []
    migration_warnings: list[str] = []

    if isinstance(value, dict) and value.get("query_ir_version") is None:
        warnings.append("legacy_query_ir_without_version_interpreted_as_v1")

    if target_version == "2.0":
        query = value if detected == "2.0" else migrate_v1_to_v2(value)  # type: ignore[arg-type]
        if detected == "1":
            migration_warnings.append(V1_TO_V2_WARNING)
        query = QueryNode.model_validate(query)
    elif target_version == "1":
        if detected == "2.0":
            query = convert_v2_to_v1(value if isinstance(value, QueryNode) else QueryNode.model_validate(value))
            migration_warnings.append(V2_TO_V1_WARNING)
        else:
            query = coerce_query_ir_v1(value)  # type: ignore[arg-type]
    else:
        raise QueryIRVersionError(f"Unsupported target QueryIR version: {target_version!r}")

    return LoadedQueryIR(
        query_ir_version=target_version,
        query_ir=query,
        diagnostics=QueryIRLoadDiagnostics(
            detected_version=detected,
            target_version=target_version,
            warnings=warnings,
            migration_warnings=migration_warnings,
        ),
    )


__all__ = [
    "LoadedQueryIR",
    "QueryIRLoadDiagnostics",
    "QueryIRVersionError",
    "detect_query_ir_version",
    "load_query_ir",
]
