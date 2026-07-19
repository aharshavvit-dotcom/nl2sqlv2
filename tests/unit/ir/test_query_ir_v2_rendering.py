"""
Purpose: Verifies ir unit behaviour consolidated from fragmented test files.
Required because: Boolean, NULL, IN and BETWEEN rendering are one QueryIR rendering responsibility.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_query_ir_v2_boolean_renderer.py
import pytest

from ir.query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer, QueryIRV2RenderingError
from ir.query_ir_v2_models import BetweenPredicate, FromItem, InLiteralPredicate, NullPredicate, QueryNode, SelectItem
from tests.query_ir_v2_boolean_helpers import and_tree, col, eq, lit, or_tree


def _query(where) -> QueryNode:
    return QueryNode(
        from_item=FromItem(table="customers"),
        select_items=[SelectItem(role="dimension", expression=col("id"), alias="id")],
        where=where,
        capability_metadata={"source_capability_labels": ["OR_FILTER"]} if "OR" in str(where.model_dump()) else {},
    )


def test_renderer_parenthesizes_or_under_and() -> None:
    query = _query(and_tree(or_tree(eq("region", "US"), eq("region", "CA")), eq("status", "ACTIVE")))

    sql = QueryIRV2NativeRenderer(enable_or_rendering=True).render(query)

    assert '("customers"."region" = ' in sql
    assert " OR " in sql
    assert ") AND " in sql


def test_renderer_default_rejects_or_until_feature_flag_enabled() -> None:
    query = _query(or_tree(eq("region", "US"), eq("region", "CA")))

    with pytest.raises(QueryIRV2RenderingError) as excinfo:
        QueryIRV2NativeRenderer().render(query)

    assert excinfo.value.code == "v2_or_rendering_disabled"


def test_renderer_renders_null_in_and_between_predicates() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        select_items=[SelectItem(role="dimension", expression=col("id"), alias="id")],
        where=and_tree(
            NullPredicate(expression=col("deleted_at")),
            InLiteralPredicate(expression=col("region"), values=[lit("US"), lit("CA")]),
            BetweenPredicate(expression=col("created_at"), lower=lit("2026-01-01", "date"), upper=lit("2026-12-31", "date")),
        ),
    )

    sql = QueryIRV2NativeRenderer(enable_or_rendering=True).render(query)

    assert '"customers"."deleted_at" IS NULL' in sql
    assert '"customers"."region" IN (' in sql
    assert '"customers"."created_at" BETWEEN ' in sql


def test_renderer_renders_not_with_parentheses() -> None:
    from ir.query_ir_v2_models import NotPredicate

    query = _query(NotPredicate(operand=or_tree(eq("region", "US"), eq("region", "CA"))))
    sql = QueryIRV2NativeRenderer(enable_or_rendering=True).render(query)

    assert "WHERE NOT ((" in sql or "WHERE NOT (" in sql
