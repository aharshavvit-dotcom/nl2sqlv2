from __future__ import annotations

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

