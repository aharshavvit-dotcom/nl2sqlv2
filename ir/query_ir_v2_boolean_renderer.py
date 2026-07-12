"""QueryIR v2 boolean renderer — backward-compatible public entry point.

All rendering logic has been decomposed into ``ir/query_ir_v2_rendering/``.
This file re-exports the public API so that all existing imports continue to work:

    from ir.query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer
    from ir.query_ir_v2_boolean_renderer import QueryIRV2RenderingError

The ``QueryIRV2NativeRenderer`` class now supports the full v2 construct set:
GROUP BY, HAVING, CASE, subqueries, windows, CTEs, set operations.
"""

from __future__ import annotations

# Re-export from the internal rendering package
from .query_ir_v2_rendering.queries import (  # noqa: F401
    QueryIRV2NativeRenderer,
    QueryIRV2RenderingError,
    _contains_or,
)

__all__ = ["QueryIRV2NativeRenderer", "QueryIRV2RenderingError"]

