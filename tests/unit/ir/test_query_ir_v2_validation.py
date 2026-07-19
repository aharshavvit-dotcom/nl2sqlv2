"""
Purpose: Verifies ir unit behaviour consolidated from fragmented test files.
Required because: Validation, canonicalization and recursive predicate limits protect one QueryIR validation contract.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_query_ir_v2_validation.py
from ir.query_ir_v2_models import (
    CaseExpression,
    ColumnExpression,
    ComparisonPredicate,
    FromItem,
    LiteralExpression,
    QueryNode,
    SelectItem,
)
from ir.query_ir_v2_validation import QueryIRV2Validator


def test_validation_rejects_duplicate_aliases() -> None:
    query = QueryNode(
        from_item=FromItem(table="orders"),
        select_items=[
            SelectItem(alias="value", expression=ColumnExpression(table="orders", column="amount")),
            SelectItem(alias="value", expression=ColumnExpression(table="orders", column="status")),
        ],
    )

    result = QueryIRV2Validator().validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "duplicate_select_alias" for issue in result.issues)


def test_validation_reports_forbidden_mutation_payload() -> None:
    result = QueryIRV2Validator().validate({"query_ir_version": "2.0", "query_type": "DELETE"})

    assert not result.is_valid
    assert any(issue.issue_type == "forbidden_mutation_query_type" for issue in result.issues)


def test_validation_reports_negative_limit_from_raw_payload() -> None:
    result = QueryIRV2Validator().validate(
        {"query_ir_version": "2.0", "from_item": {"from_type": "TABLE", "table": "orders"}, "limit": -1}
    )

    assert not result.is_valid
    assert any("non-negative" in issue.message for issue in result.issues)


def test_renderer_support_validation_flags_advanced_nodes() -> None:
    query = QueryNode(
        from_item=FromItem(table="orders"),
        select_items=[
            SelectItem(
                alias="bucket",
                expression=CaseExpression(
                    cases=[
                        {
                            "when": ComparisonPredicate(
                                left=ColumnExpression(table="orders", column="amount"),
                                operator=">",
                                right=LiteralExpression(value=100, value_type="number"),
                            ),
                            "then": LiteralExpression(value="large"),
                        }
                    ],
                    else_expression=LiteralExpression(value="small"),
                ),
            )
        ],
    )

    result = QueryIRV2Validator(enforce_renderer_support=True).validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "unsupported_v2_rendering_capability" for issue in result.issues)


# Source: tests/test_query_ir_v2_depth_limits.py
from ir.query_ir_v2_models import ColumnExpression, FromItem, QueryNode, SelectItem, UnaryOperationExpression
from ir.query_ir_v2_validation import QueryIRV2Validator


def test_query_ir_v2_depth_limit_blocks_deep_recursive_expression() -> None:
    expression = ColumnExpression(table="orders", column="amount")
    for _ in range(6):
        expression = UnaryOperationExpression(operator="-", operand=expression)
    query = QueryNode(
        from_item=FromItem(table="orders"),
        select_items=[SelectItem(alias="amount", expression=expression)],
    )

    result = QueryIRV2Validator(max_recursive_depth=5).validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "recursive_depth_exceeded" for issue in result.issues)


def test_query_ir_v2_depth_limit_allows_shallow_expression() -> None:
    query = QueryNode(
        from_item=FromItem(table="orders"),
        select_items=[SelectItem(alias="amount", expression=ColumnExpression(table="orders", column="amount"))],
    )

    result = QueryIRV2Validator(max_recursive_depth=5).validate(query)

    assert result.is_valid


# Source: tests/test_query_ir_v2_boolean_validation.py
from ir.query_ir_v2_models import BetweenPredicate, FromItem, InLiteralPredicate, QueryNode
from ir.query_ir_v2_validation import QueryIRV2Validator
from tests.query_ir_v2_boolean_helpers import col, eq, lit, or_tree


def test_boolean_validation_accepts_or_when_capability_label_matches() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=or_tree(eq("region", "US"), eq("region", "CA")),
        capability_metadata={"source_capability_labels": ["OR_FILTER"]},
    )

    assert QueryIRV2Validator().validate(query).is_valid


def test_boolean_validation_rejects_incompatible_comparison() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=eq("created_at", "not-a-date"),
    )
    query.where.operator = ">"  # type: ignore[union-attr]

    result = QueryIRV2Validator().validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "incompatible_comparison_operands" for issue in result.issues)


def test_boolean_validation_rejects_mixed_in_literal_types() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=InLiteralPredicate(expression=col("region"), values=[lit("US"), lit(10, "number")]),
    )

    result = QueryIRV2Validator().validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "incompatible_in_literal_values" for issue in result.issues)


def test_boolean_validation_rejects_between_incompatible_bounds() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=BetweenPredicate(expression=col("created_at"), lower=lit("2026-01-01", "date"), upper=lit(10, "number")),
    )

    result = QueryIRV2Validator().validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "incompatible_between_bounds" for issue in result.issues)


# Source: tests/test_query_ir_v2_boolean_depth_limits.py
from ir.query_ir_v2_models import FromItem, NotPredicate, QueryNode
from ir.query_ir_v2_validation import QueryIRV2Validator
from tests.query_ir_v2_boolean_helpers import and_tree, eq


def test_boolean_depth_limit_blocks_deep_not_chain() -> None:
    predicate = eq("region", "US")
    for _ in range(6):
        predicate = NotPredicate(operand=predicate)
    query = QueryNode(from_item=FromItem(table="customers"), where=predicate)

    result = QueryIRV2Validator(max_recursive_depth=5).validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "recursive_depth_exceeded" for issue in result.issues)


def test_boolean_node_count_limit_blocks_large_tree() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=and_tree(eq("region", "US"), eq("status", "ACTIVE"), eq("tier", "GOLD")),
    )

    result = QueryIRV2Validator(max_predicate_nodes=2).validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "predicate_node_count_exceeded" for issue in result.issues)


# Source: tests/test_query_ir_v2_boolean_canonicalization.py
from ir.query_ir_v2_boolean_canonicalization import canonicalize_predicate
from ir.query_ir_v2_models import BooleanPredicate, NotPredicate
from tests.query_ir_v2_boolean_helpers import and_tree, eq, or_tree


def test_canonicalization_flattens_nested_same_operator_and_is_idempotent() -> None:
    predicate = and_tree(eq("region", "US"), and_tree(eq("status", "ACTIVE"), eq("tier", "GOLD")))

    once = canonicalize_predicate(predicate)
    twice = canonicalize_predicate(once)

    assert isinstance(once, BooleanPredicate)
    assert once.operator == "AND"
    assert len(once.operands) == 3
    assert once.model_dump() == twice.model_dump()


def test_canonicalization_preserves_mixed_and_or_grouping() -> None:
    predicate = and_tree(or_tree(eq("region", "US"), eq("region", "CA")), eq("status", "ACTIVE"))
    canonical = canonicalize_predicate(predicate)

    assert isinstance(canonical, BooleanPredicate)
    assert isinstance(canonical.operands[0], BooleanPredicate)
    assert canonical.operands[0].operator == "OR"


def test_canonicalization_preserves_not_nodes() -> None:
    predicate = NotPredicate(operand=or_tree(eq("region", "US"), eq("region", "CA")))
    canonical = canonicalize_predicate(predicate)

    assert isinstance(canonical, NotPredicate)
    assert isinstance(canonical.operand, BooleanPredicate)
    assert canonical.operand.operator == "OR"
