"""
Purpose: Verifies ir unit behaviour consolidated from fragmented test files.
Required because: QueryIR v2 model, literal, serialization and fingerprint contracts belong to the same model responsibility.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_query_ir_v2_models.py
import pytest
from pydantic import ValidationError

from ir.query_ir_v2_models import (
    AggregationExpression,
    BinaryOperationExpression,
    ColumnExpression,
    FromItem,
    LiteralExpression,
    QueryNode,
    SelectItem,
)


def test_query_ir_v2_requires_explicit_version() -> None:
    query = QueryNode(
        query_ir_id="qir-v2",
        intent="metric_summary",
        from_item=FromItem(table="orders"),
        select_items=[
            SelectItem(
                role="metric",
                alias="revenue",
                expression=AggregationExpression(
                    function="sum",
                    argument=ColumnExpression(table="orders", column="amount"),
                ),
            )
        ],
    )

    assert query.query_ir_version == "2.0"
    assert query.select_items[0].expression.expression_type == "AGGREGATION"


def test_recursive_expression_discriminated_union_roundtrips_from_dict() -> None:
    query = QueryNode.model_validate(
        {
            "query_ir_version": "2.0",
            "from_item": {"from_type": "TABLE", "table": "order_items"},
            "select_items": [
                {
                    "role": "metric",
                    "alias": "revenue",
                    "expression": {
                        "expression_type": "AGGREGATION",
                        "function": "SUM",
                        "argument": {
                            "expression_type": "BINARY_OPERATION",
                            "operator": "*",
                            "left": {"expression_type": "COLUMN", "table": "order_items", "column": "quantity"},
                            "right": {"expression_type": "COLUMN", "table": "order_items", "column": "price"},
                        },
                    },
                }
            ],
        }
    )

    aggregation = query.select_items[0].expression
    assert isinstance(aggregation, AggregationExpression)
    assert isinstance(aggregation.argument, BinaryOperationExpression)


def test_unknown_v2_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryNode.model_validate({"query_ir_version": "3.0", "from_item": {"from_type": "TABLE", "table": "orders"}})


def test_literal_expression_rejects_untyped_dict_and_accepts_strict_types() -> None:
    """Strict literal types: no arbitrary Python objects (correction #4)."""
    # Dict values are rejected
    with pytest.raises(ValidationError):
        LiteralExpression(value={"start": "2026-01-01"}, value_type="object")

    # Proper DATE literal uses string representation
    from ir.query_ir_v2_models import LiteralValueType
    date_lit = LiteralExpression(value="2026-01-01", value_type=LiteralValueType.DATE, source_text="'2026-01-01'")
    assert date_lit.value == "2026-01-01"
    assert date_lit.value_type == LiteralValueType.DATE
    assert date_lit.source_text == "'2026-01-01'"

    # DECIMAL preserves precision as string
    dec_lit = LiteralExpression(value="1234567890.123400", value_type=LiteralValueType.DECIMAL)
    assert dec_lit.value == "1234567890.123400"

    # Legacy lowercase coercion
    legacy = LiteralExpression(value=42, value_type="integer")
    assert legacy.value_type == LiteralValueType.INTEGER

    # Float coerces to DECIMAL
    float_lit = LiteralExpression(value="3.14", value_type="float")
    assert float_lit.value_type == LiteralValueType.DECIMAL


# Source: tests/test_query_ir_v2_boolean_models.py
from ir.query_ir_v2_models import BooleanPredicate, InLiteralPredicate, NotPredicate, NullPredicate
from tests.query_ir_v2_boolean_helpers import col, eq, lit, or_tree


def test_boolean_predicate_tree_uses_phase2b_node_names() -> None:
    predicate = or_tree(eq("region", "US"), eq("region", "CA"))

    payload = predicate.model_dump()

    assert payload["predicate_type"] == "BOOLEAN_PREDICATE"
    assert payload["operator"] == "OR"
    assert payload["operands"][0]["predicate_type"] == "COMPARISON_PREDICATE"


def test_not_null_and_in_literal_predicate_nodes() -> None:
    not_predicate = NotPredicate(operand=eq("status", "ACTIVE"))
    null_predicate = NullPredicate(expression=col("deleted_at"), negated=True)
    in_predicate = InLiteralPredicate(expression=col("region"), values=[lit("US"), lit("CA")])

    assert not_predicate.predicate_type == "NOT_PREDICATE"
    assert null_predicate.predicate_type == "NULL_PREDICATE"
    assert in_predicate.predicate_type == "IN_LITERAL_PREDICATE"


def test_boolean_predicate_rejects_single_operand() -> None:
    try:
        BooleanPredicate(operator="AND", operands=[eq("region", "US")])
    except ValueError as exc:
        assert "requires at least 2" in str(exc)
    else:
        raise AssertionError("single operand BooleanPredicate should be invalid")


# Source: tests/test_query_ir_v2_serialization.py
from ir.query_ir_migration import migrate_v1_to_v2
from ir.query_ir_v2_serialization import canonical_query_ir_v2_dict, dumps_query_ir_v2, loads_query_ir_v2
from tests.query_ir_v2_test_helpers import make_v1_metric_summary


def test_query_ir_v2_serialization_is_deterministic() -> None:
    v2 = migrate_v1_to_v2(make_v1_metric_summary())

    assert dumps_query_ir_v2(v2) == dumps_query_ir_v2(v2.model_dump())
    assert '"query_ir_version":"2.0"' in dumps_query_ir_v2(v2)


def test_query_ir_v2_deserializes_to_typed_model() -> None:
    payload = dumps_query_ir_v2(migrate_v1_to_v2(make_v1_metric_summary()))
    restored = loads_query_ir_v2(payload)

    assert restored.query_ir_version == "2.0"
    assert restored.select_items[0].alias == "revenue"


def test_canonical_dict_orders_nested_fields_stably() -> None:
    v2 = migrate_v1_to_v2(make_v1_metric_summary())
    canonical = canonical_query_ir_v2_dict(v2)

    assert list(canonical) == sorted(canonical)
    assert list(canonical["metadata"]) == sorted(canonical["metadata"])


# Source: tests/test_query_ir_v2_fingerprint.py
from ir.query_ir_migration import migrate_v1_to_v2
from ir.query_ir_v2_serialization import fingerprint_query_ir_v2
from tests.query_ir_v2_test_helpers import make_v1_metric_summary


def test_query_ir_v2_fingerprint_is_stable_for_equivalent_payloads() -> None:
    v2 = migrate_v1_to_v2(make_v1_metric_summary())

    assert fingerprint_query_ir_v2(v2) == fingerprint_query_ir_v2(v2.model_dump())


def test_query_ir_v2_fingerprint_changes_when_versioned_payload_changes() -> None:
    v2 = migrate_v1_to_v2(make_v1_metric_summary())
    changed = v2.model_copy(update={"intent": "count_records"})

    assert fingerprint_query_ir_v2(v2) != fingerprint_query_ir_v2(changed)
    assert v2.query_ir_version == "2.0"
