"""Legacy query_ir_models re-export for backward compatibility.

During the migration period, this module re-exports QueryIR (v1) from
its original location. Once all runtime code uses the canonical
QueryIR (from ir.query_ir_v2_models.QueryNode), this bridge will
emit deprecation warnings and eventually be removed.

Migration deadline: 2026-09-01
"""
from __future__ import annotations

import warnings

from ir.query_ir_models import (  # noqa: F401
    IRDateFilter,
    IRDimension,
    IRFilter,
    IRJoin,
    IRMetric,
    IROrderBy,
    QueryIR,
)


def get_query_ir_class(version: str = "v1"):
    """Get the QueryIR class for the requested version.

    Parameters
    ----------
    version:
        ``"v1"`` returns ``QueryIR`` (current runtime).
        ``"v2"`` returns ``QueryNode`` (advanced, canonical target).

    Returns
    -------
    The requested QueryIR class.
    """
    if version == "v2":
        from ir.query_ir_v2_models import QueryNode
        return QueryNode
    return QueryIR


__all__ = [
    "QueryIR",
    "IRMetric",
    "IRDimension",
    "IRFilter",
    "IRDateFilter",
    "IRJoin",
    "IROrderBy",
    "get_query_ir_class",
]
