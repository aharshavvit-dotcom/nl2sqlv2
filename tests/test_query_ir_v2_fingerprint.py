from __future__ import annotations

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
