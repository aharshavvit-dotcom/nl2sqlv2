import pytest
import datetime
from pathlib import Path
from nl2sqlv2.model_bundle.bundle_validator import BundleValidator
from nl2sqlv2.model_bundle.bundle_manifest import BundleManifest, TrainingPipelineMetrics

def test_pipeline_run_id_alone_fails_identity(tmp_path):
    validator = BundleValidator(str(tmp_path))
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        commit_sha="abcdef",
        training_metrics=TrainingPipelineMetrics(pipeline_run_id="run-456")
    )
    report = {
        "pipeline_run_id": "run-456",
        # Missing bundle_id and candidate_bundle_dir
    }
    issues = validator._verify_report_identity(report, manifest)
    assert len(issues) > 0
    assert any("identity token" in issue.lower() for issue in issues)

def test_bundle_id_matching(tmp_path):
    validator = BundleValidator(str(tmp_path))
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        commit_sha="abcdef",
    )
    report = {
        "bundle_id": "bundle-123",
    }
    issues = validator._verify_report_identity(report, manifest)
    assert len(issues) == 0

def test_candidate_bundle_dir_matching(tmp_path):
    validator = BundleValidator(str(tmp_path))
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        commit_sha="abcdef",
    )
    report = {
        "candidate_bundle_dir": str(tmp_path),
    }
    issues = validator._verify_report_identity(report, manifest)
    assert len(issues) == 0

def test_stale_commit_sha_fails(tmp_path):
    validator = BundleValidator(str(tmp_path))
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        commit_sha="abcdef",
    )
    report = {
        "bundle_id": "bundle-123",
        "commit_sha": "123456" # Mismatched
    }
    issues = validator._verify_report_identity(report, manifest)
    assert len(issues) > 0
    assert any("commit" in issue.lower() for issue in issues)

def test_stale_generated_at_fails(tmp_path):
    validator = BundleValidator(str(tmp_path))
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        commit_sha="abcdef",
    )
    # Generated way before the bundle was created
    report = {
        "bundle_id": "bundle-123",
        "generated_at": "2020-01-01T00:00:00Z"
    }
    issues = validator._verify_report_identity(report, manifest)
    assert len(issues) > 0
    assert any("stale" in issue.lower() or "generated" in issue.lower() for issue in issues)
