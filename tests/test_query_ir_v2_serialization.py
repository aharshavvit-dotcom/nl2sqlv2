from __future__ import annotations

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
