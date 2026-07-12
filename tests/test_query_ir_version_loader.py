from __future__ import annotations

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
