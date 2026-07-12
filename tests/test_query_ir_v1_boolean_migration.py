from __future__ import annotations

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
