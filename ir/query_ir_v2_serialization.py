from __future__ import annotations

import hashlib
import json
from typing import Any

from .query_ir_v2_models import QueryNode


def coerce_query_ir_v2(value: QueryNode | dict[str, Any] | str | bytes) -> QueryNode:
    if isinstance(value, QueryNode):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"Expected QueryIR v2 object, dict, JSON str, or bytes; got {type(value).__name__}")
    return QueryNode.model_validate(value)


def canonical_query_ir_v2_dict(value: QueryNode | dict[str, Any] | str | bytes) -> dict[str, Any]:
    query = coerce_query_ir_v2(value)
    return _canonicalize(query.model_dump(mode="json", exclude_none=True))


def dumps_query_ir_v2(value: QueryNode | dict[str, Any] | str | bytes, *, indent: int | None = None) -> str:
    payload = canonical_query_ir_v2_dict(value)
    if indent is None:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=indent)


def loads_query_ir_v2(value: str | bytes) -> QueryNode:
    return coerce_query_ir_v2(value)


def fingerprint_query_ir_v2(value: QueryNode | dict[str, Any] | str | bytes, *, algorithm: str = "sha256") -> str:
    payload = dumps_query_ir_v2(value)
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported fingerprint algorithm: {algorithm}") from exc
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


__all__ = [
    "canonical_query_ir_v2_dict",
    "coerce_query_ir_v2",
    "dumps_query_ir_v2",
    "fingerprint_query_ir_v2",
    "loads_query_ir_v2",
]
