from __future__ import annotations

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
