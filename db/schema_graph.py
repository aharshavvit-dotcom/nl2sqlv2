"""Compatibility bridge for nl2sql_v1.schema -> db.schema_graph migration.

This module re-exports all public names from nl2sql_v1.schema under
the canonical db.schema_graph location. All new code should import
from db.schema_graph; this compat module exists only to support the
~50 existing imports during the migration window.

Migration deadline: 2026-09-01
"""
from __future__ import annotations

import warnings

from nl2sql_v1.schema import (  # noqa: F401
    ColumnInfo,
    ForeignKeyInfo,
    SchemaGraph,
    TableInfo,
)

__all__ = [
    "SchemaGraph",
    "TableInfo",
    "ColumnInfo",
    "ForeignKeyInfo",
]
