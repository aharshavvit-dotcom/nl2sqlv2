from __future__ import annotations

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
