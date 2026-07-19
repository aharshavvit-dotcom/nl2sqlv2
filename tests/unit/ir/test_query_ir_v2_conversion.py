"""
Purpose: Verifies ir unit behaviour consolidated from fragmented test files.
Required because: SQL-to-QueryIR conversion belongs in the canonical conversion module.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_sql_to_query_ir_v2_boolean_conversion.py
from ir.query_ir_v2_models import BooleanPredicate, NotPredicate, NullPredicate
from ir.sql_to_query_ir_v2 import SQLToQueryIRV2Converter


def test_sql_to_v2_converts_simple_or() -> None:
    query = SQLToQueryIRV2Converter().convert("SELECT id FROM customers WHERE region = 'US' OR region = 'CA'")

    assert isinstance(query.where, BooleanPredicate)
    assert query.where.operator == "OR"
    assert "OR_FILTER" in query.capability_metadata.source_capability_labels


def test_sql_to_v2_preserves_or_under_and_grouping() -> None:
    query = SQLToQueryIRV2Converter().convert(
        "SELECT id FROM customers WHERE (region = 'US' OR region = 'CA') AND status = 'ACTIVE'"
    )

    assert isinstance(query.where, BooleanPredicate)
    assert query.where.operator == "AND"
    assert isinstance(query.where.operands[0], BooleanPredicate)
    assert query.where.operands[0].operator == "OR"


def test_sql_to_v2_preserves_and_under_or_grouping() -> None:
    query = SQLToQueryIRV2Converter().convert(
        "SELECT id FROM customers WHERE region = 'US' OR (region = 'CA' AND status = 'ACTIVE')"
    )

    assert isinstance(query.where, BooleanPredicate)
    assert query.where.operator == "OR"
    assert isinstance(query.where.operands[1], BooleanPredicate)
    assert query.where.operands[1].operator == "AND"


def test_sql_to_v2_converts_not_over_or() -> None:
    query = SQLToQueryIRV2Converter().convert(
        "SELECT id FROM customers WHERE NOT (region = 'US' OR region = 'CA')"
    )

    assert isinstance(query.where, NotPredicate)
    assert isinstance(query.where.operand, BooleanPredicate)
    assert query.where.operand.operator == "OR"


def test_sql_to_v2_converts_null_predicates() -> None:
    query = SQLToQueryIRV2Converter().convert(
        "SELECT id FROM customers WHERE deleted_at IS NULL OR status IS NOT NULL"
    )

    assert isinstance(query.where, BooleanPredicate)
    assert all(isinstance(item, NullPredicate) for item in query.where.operands)
    assert query.where.operands[0].negated is False
    assert query.where.operands[1].negated is True
