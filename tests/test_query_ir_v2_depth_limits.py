from __future__ import annotations

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
