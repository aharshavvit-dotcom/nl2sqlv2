"""Tests for run-scoped pipeline state and report identity.

Validates:
- Run state creation and identity
- Row accounting fields
- Promotion blocker detection
- Report envelope generation
- Artifact recording and checksums
- Save/load round-trip
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from evaluation.pipeline_state import (
    PipelineRunState,
    create_run_state,
    compute_file_sha256,
)
from evaluation.report_schemas import (
    RowAccounting,
    PromotionEligibility,
    CheckpointIdentity,
    ReportIdentity,
    REPORT_SCHEMA_VERSION,
)


class TestPipelineRunState:
    def test_create_has_run_id(self):
        state = create_run_state(pipeline_name="test_pipeline")
        assert state.run_id.startswith("run-")
        assert len(state.run_id) > 10
        assert state.pipeline_name == "test_pipeline"

    def test_explicit_run_id(self):
        state = create_run_state(pipeline_run_id="custom-id-123")
        assert state.run_id == "custom-id-123"

    def test_report_identity(self):
        state = create_run_state(
            pipeline_name="test",
            pipeline_run_id="run-abc",
        )
        assert state.report_identity.report_type == "pipeline_run"
        assert state.report_identity.pipeline_run_id == "run-abc"
        assert state.report_identity.report_schema_version == REPORT_SCHEMA_VERSION

    def test_row_accounting_default(self):
        state = create_run_state()
        assert state.row_accounting.total_rows_evaluated == 0
        assert state.row_accounting.standard_rows_evaluated == 0

    def test_row_accounting_update(self):
        state = create_run_state()
        state.row_accounting = RowAccounting(
            standard_rows_evaluated=292,
            unseen_db_rows_evaluated=601,
            total_rows_evaluated=893,
            standard_predictions_generated=292,
            unseen_db_predictions_generated=601,
            total_predictions_generated=893,
        )
        assert state.row_accounting.total_rows_evaluated == 893
        assert state.row_accounting.standard_rows_evaluated == 292

    def test_promotion_blockers(self):
        state = create_run_state(
            model_artifact_source="artifact_dirs",
            full_bundle_runtime_used=False,
        )
        state.promotion = PromotionEligibility(
            eligible_for_promotion=False,
            evaluation_scope="artifact_dirs",
            full_bundle_runtime_used=False,
            promotion_blockers=[
                "model_artifact_source_is_artifact_dirs",
                "full_bundle_runtime_not_used",
            ],
        )
        assert state.promotion.eligible_for_promotion is False
        assert len(state.promotion.promotion_blockers) == 2

    def test_report_envelope(self):
        state = create_run_state(pipeline_run_id="run-envelope-test")
        envelope = state.to_report_envelope()
        assert envelope["pipeline_run_id"] == "run-envelope-test"
        assert "report_schema_version" in envelope
        assert "metric_definitions_version" in envelope
        assert "pipeline_run_state" in envelope

    def test_record_artifact(self):
        state = create_run_state()
        state.record_artifact("path/to/report.json", "sha256-abc123")
        assert "path/to/report.json" in state.artifacts_produced
        assert state.artifact_checksums["path/to/report.json"] == "sha256-abc123"

    def test_record_duplicate_artifact(self):
        state = create_run_state()
        state.record_artifact("path/to/report.json")
        state.record_artifact("path/to/report.json")
        assert state.artifacts_produced.count("path/to/report.json") == 1

    def test_record_quality_gate(self):
        state = create_run_state()
        state.record_quality_gate({
            "passed": False,
            "quality_gate_mode": "production",
            "failed_checks": [{"metric": "sql_validation_rate", "actual": 0.8, "expected": 0.9}],
        })
        assert state.quality_gate_passed is False
        assert state.quality_gate_mode == "production"
        assert len(state.quality_gate_failures) == 1

    def test_complete(self):
        state = create_run_state()
        assert state.completed_at is None
        state.complete()
        assert state.completed_at is not None


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path):
        state = create_run_state(
            pipeline_name="round_trip_test",
            pipeline_run_id="run-round-trip",
            model_artifact_source="model_bundle",
            evaluation_mode="real_model_predictions",
            full_bundle_runtime_used=True,
        )
        state.row_accounting = RowAccounting(
            standard_rows_evaluated=100,
            unseen_db_rows_evaluated=50,
            total_rows_evaluated=150,
        )
        state.record_artifact("artifact.json", "sha256-hash")
        state.complete()

        save_path = tmp_path / "run_state.json"
        state.save(save_path)
        assert save_path.exists()

        loaded = PipelineRunState.load(save_path)
        assert loaded.run_id == "run-round-trip"
        assert loaded.pipeline_name == "round_trip_test"
        assert loaded.row_accounting.total_rows_evaluated == 150
        assert loaded.completed_at is not None
        assert "artifact.json" in loaded.artifacts_produced


class TestFileChecksum:
    def test_compute_sha256(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        checksum = compute_file_sha256(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length

    def test_different_content_different_hash(self, tmp_path: Path):
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("content_a", encoding="utf-8")
        file_b.write_text("content_b", encoding="utf-8")
        assert compute_file_sha256(file_a) != compute_file_sha256(file_b)
