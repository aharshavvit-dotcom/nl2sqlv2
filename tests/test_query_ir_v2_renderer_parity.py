from __future__ import annotations

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
