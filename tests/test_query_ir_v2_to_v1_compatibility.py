from __future__ import annotations

import pytest

from ir.query_ir_migration import QueryIRCompatibilityError, convert_v2_to_v1, migrate_v1_to_v2
from ir.query_ir_v2_models import CaseExpression, ColumnExpression, FromItem, LiteralExpression, QueryNode, SelectItem
from tests.query_ir_v2_test_helpers import make_v1_filter, make_v1_product_revenue


def test_v2_to_v1_restores_representable_subset() -> None:
    original = make_v1_filter()
    restored = convert_v2_to_v1(migrate_v1_to_v2(original))

    assert restored.intent == original.intent
    assert restored.base_table == original.base_table
    assert restored.filters[0].operator == "equals"
    assert restored.filters[0].value == "completed"


def test_v2_to_v1_preserves_binary_metric_expression_for_renderer() -> None:
    restored = convert_v2_to_v1(migrate_v1_to_v2(make_v1_product_revenue()))

    assert restored.metrics[0].expression == "order_items.quantity * order_items.price"
    assert restored.metrics[0].column is None


def test_advanced_v2_node_returns_typed_compatibility_error() -> None:
    query = QueryNode(
        from_item=FromItem(table="orders"),
        select_items=[
            SelectItem(
                alias="bucket",
                expression=CaseExpression(
                    cases=[],
                    else_expression=LiteralExpression(value="small"),
                ),
            )
        ],
    )

    with pytest.raises(QueryIRCompatibilityError) as excinfo:
        convert_v2_to_v1(query)

    assert excinfo.value.code == "unsupported_v2_rendering_capability"
    assert excinfo.value.capability == "case_expression"


def test_manual_simple_v2_can_convert_to_v1() -> None:
    query = QueryNode(
        query_ir_id="manual",
        intent="show_records",
        template_id="show_records",
        from_item=FromItem(table="users"),
        required_tables=["users"],
        select_items=[SelectItem(role="dimension", alias="name", expression=ColumnExpression(table="users", column="name"))],
    )

    restored = convert_v2_to_v1(query)

    assert restored.base_table == "users"
    assert restored.dimensions[0].column == "name"
