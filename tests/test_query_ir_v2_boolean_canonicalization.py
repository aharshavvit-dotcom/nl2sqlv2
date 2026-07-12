from __future__ import annotations

from ir.query_ir_v2_boolean_canonicalization import canonicalize_predicate
from ir.query_ir_v2_models import BooleanPredicate, NotPredicate
from tests.query_ir_v2_boolean_helpers import and_tree, eq, or_tree


def test_canonicalization_flattens_nested_same_operator_and_is_idempotent() -> None:
    predicate = and_tree(eq("region", "US"), and_tree(eq("status", "ACTIVE"), eq("tier", "GOLD")))

    once = canonicalize_predicate(predicate)
    twice = canonicalize_predicate(once)

    assert isinstance(once, BooleanPredicate)
    assert once.operator == "AND"
    assert len(once.operands) == 3
    assert once.model_dump() == twice.model_dump()


def test_canonicalization_preserves_mixed_and_or_grouping() -> None:
    predicate = and_tree(or_tree(eq("region", "US"), eq("region", "CA")), eq("status", "ACTIVE"))
    canonical = canonicalize_predicate(predicate)

    assert isinstance(canonical, BooleanPredicate)
    assert isinstance(canonical.operands[0], BooleanPredicate)
    assert canonical.operands[0].operator == "OR"


def test_canonicalization_preserves_not_nodes() -> None:
    predicate = NotPredicate(operand=or_tree(eq("region", "US"), eq("region", "CA")))
    canonical = canonicalize_predicate(predicate)

    assert isinstance(canonical, NotPredicate)
    assert isinstance(canonical.operand, BooleanPredicate)
    assert canonical.operand.operator == "OR"
