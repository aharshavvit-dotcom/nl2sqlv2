from __future__ import annotations

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
