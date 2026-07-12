from __future__ import annotations

from ir.query_ir_migration import migrate_v1_to_v2
from ir.query_ir_v2_models import AggregationExpression, BinaryOperationExpression
from ir.query_ir_v2_serialization import dumps_query_ir_v2
from tests.query_ir_v2_test_helpers import make_v1_metric_by_dimension, make_v1_product_revenue


def test_v1_to_v2_maps_core_slots() -> None:
    v1 = make_v1_metric_by_dimension()
    v2 = migrate_v1_to_v2(v1)

    assert v2.query_ir_version == "2.0"
    assert v2.from_item and v2.from_item.table == "orders"
    assert v2.required_tables == ["orders", "customers"]
    assert len(v2.joins) == 1
    assert len(v2.predicates) == 0
    assert {item.role for item in v2.select_items} == {"dimension", "metric"}
    assert v2.metadata["migration"]["from_query_ir_version"] == "1"


def test_v1_to_v2_migration_is_idempotent_for_v2_input() -> None:
    first = migrate_v1_to_v2(make_v1_metric_by_dimension())
    second = migrate_v1_to_v2(first)

    assert dumps_query_ir_v2(first) == dumps_query_ir_v2(second)


def test_v1_binary_metric_expression_becomes_recursive_v2_expression() -> None:
    v2 = migrate_v1_to_v2(make_v1_product_revenue())
    metric = next(item for item in v2.select_items if item.role == "metric")

    assert isinstance(metric.expression, AggregationExpression)
    assert isinstance(metric.expression.argument, BinaryOperationExpression)
    assert metric.expression.argument.operator == "*"
