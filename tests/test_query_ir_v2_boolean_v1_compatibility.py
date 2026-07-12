from __future__ import annotations

import pytest

from ir.query_ir_migration import QueryIRCompatibilityError, convert_v2_to_v1
from ir.query_ir_v2_models import FromItem, NotPredicate, QueryNode
from tests.query_ir_v2_boolean_helpers import and_tree, eq, or_tree


def test_and_only_boolean_tree_converts_to_v1_filters() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=and_tree(eq("region", "US"), eq("status", "ACTIVE")),
        required_tables=["customers"],
    )

    v1 = convert_v2_to_v1(query)

    assert [item.column for item in v1.filters] == ["region", "status"]
    assert [item.value for item in v1.filters] == ["US", "ACTIVE"]


def test_or_boolean_tree_rejects_v1_compatibility() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=or_tree(eq("region", "US"), eq("region", "CA")),
        capability_metadata={"source_capability_labels": ["OR_FILTER"]},
    )

    with pytest.raises(QueryIRCompatibilityError) as excinfo:
        convert_v2_to_v1(query)

    assert excinfo.value.code == "v2_predicate_not_representable_in_v1"


def test_not_boolean_tree_rejects_v1_compatibility() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=NotPredicate(operand=eq("status", "ACTIVE")),
    )

    with pytest.raises(QueryIRCompatibilityError) as excinfo:
        convert_v2_to_v1(query)

    assert excinfo.value.code == "v2_predicate_not_representable_in_v1"
