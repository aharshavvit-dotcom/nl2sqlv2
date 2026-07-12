from __future__ import annotations

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
