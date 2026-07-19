"""
Purpose: Verifies ir unit behaviour consolidated from fragmented test files.
Required because: V1/V2 migration, compatibility and version loading form one compatibility contract.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_query_ir_v1_to_v2_migration.py
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


# Source: tests/test_query_ir_v1_boolean_migration.py
from ir.query_ir_migration import migrate_v1_to_v2
from ir.query_ir_v2_models import BooleanPredicate, ComparisonPredicate
from tests.query_ir_v2_test_helpers import make_v1_filter


def test_single_v1_filter_migrates_to_direct_predicate_root() -> None:
    v2 = migrate_v1_to_v2(make_v1_filter())

    assert isinstance(v2.where, ComparisonPredicate)
    assert len(v2.predicates) == 1


def test_multiple_v1_filters_migrate_to_and_predicate_tree() -> None:
    v1 = make_v1_filter()
    v1.filters.append(v1.filters[0].model_copy(update={"column": "region", "expression": "orders.region", "value": "US"}))

    v2 = migrate_v1_to_v2(v1)

    assert isinstance(v2.where, BooleanPredicate)
    assert v2.where.operator == "AND"
    assert len(v2.where.operands) == 2


# Source: tests/test_query_ir_v2_to_v1_compatibility.py
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


# Source: tests/test_query_ir_v2_boolean_v1_compatibility.py
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


# Source: tests/test_query_ir_version_loader.py
import pytest

from ir.query_ir_version_loader import QueryIRVersionError, detect_query_ir_version, load_query_ir
from tests.query_ir_v2_test_helpers import make_v1_metric_summary


def test_loader_detects_legacy_v1_without_version_and_migrates_to_v2() -> None:
    payload = make_v1_metric_summary().model_dump()
    loaded = load_query_ir(payload, target_version="2.0")

    assert loaded.query_ir_version == "2.0"
    assert loaded.diagnostics.detected_version == "1"
    assert "legacy_query_ir_without_version_interpreted_as_v1" in loaded.diagnostics.warnings
    assert loaded.query_ir.query_ir_version == "2.0"


def test_loader_detects_explicit_v2_and_converts_to_v1_subset() -> None:
    v2 = load_query_ir(make_v1_metric_summary().model_dump(), target_version="2.0").query_ir
    loaded = load_query_ir(v2.model_dump(), target_version="1")

    assert loaded.query_ir_version == "1"
    assert loaded.query_ir.intent == "metric_summary"
    assert loaded.diagnostics.migration_warnings


def test_loader_rejects_unknown_version() -> None:
    with pytest.raises(QueryIRVersionError):
        detect_query_ir_version({"query_ir_version": "9.9"})


# Source: tests/test_query_ir_v2_renderer_parity.py
from ir.query_ir_v2_parity import run_query_ir_v2_renderer_parity
from tests.query_ir_v2_test_helpers import supported_v1_examples


def test_v2_compatibility_adapter_preserves_supported_v1_renderer_semantics() -> None:
    report = run_query_ir_v2_renderer_parity(supported_v1_examples())

    assert {key: report[key] for key in [
        "total_migrated",
        "total_parity_passed",
        "total_migration_failures",
        "total_sql_normalization_differences",
        "unsupported_conversion_count",
    ]} == {
        "total_migrated": 6,
        "total_parity_passed": 6,
        "total_migration_failures": 0,
        "total_sql_normalization_differences": 0,
        "unsupported_conversion_count": 0,
    }
    assert report["failures"] == []
