from __future__ import annotations

import json
from pathlib import Path
import pytest

from model_bundle.bundle_validator import ModelBundleValidator, _verify_report_identity
from model_bundle.bundle_manifest import BundleManifest, load_manifest, save_manifest
from orchestration.pipeline_config import PipelineConfig
from orchestration.step_runner import (
    StepRunner,
    _predicted_sql_bundle_identity,
)


def test_required_predicted_sql_step_fails_without_candidate_manifest(tmp_path, monkeypatch):
    import orchestration.step_runner as step_runner

    monkeypatch.setattr(step_runner, "ROOT", tmp_path)
    config = PipelineConfig(
        pipeline_name="run-1",
        training={"_integrated_config": {
            "execution_aware": {"controlled_predicted_sql": {
                "enabled": True,
                "required_for_full_training": True,
            }},
            "paths": {"candidate_bundle_dir": "candidate"},
        }},
        artifacts={"evaluation_dir": str(tmp_path / "evaluation")},
    )
    result = StepRunner().run_step("run_controlled_predicted_sql_evaluation", config)
    assert result["status"] == "failed"
    assert result["error"] == "candidate_manifest_missing_for_predicted_sql"
    assert result["summary"]["identity_strength"] == "weak"


def test_required_predicted_sql_identity_fails_without_manifest_bundle_id(tmp_path):
    (tmp_path / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    result = _predicted_sql_bundle_identity(tmp_path, "run-1", required=True)
    assert result["error"] == "candidate_manifest_bundle_id_missing_for_predicted_sql"


def test_optional_predicted_sql_identity_records_weak_fallback_warning(tmp_path):
    result = _predicted_sql_bundle_identity(tmp_path, "run-1", required=False)
    assert result["bundle_id"] == "run-1"
    assert result["bundle_id_source"] == "pipeline_name_fallback"
    assert result["identity_strength"] == "weak"
    assert result["candidate_manifest_missing"] is True
    assert result["warnings"] == ["candidate_manifest_missing_for_predicted_sql"]


def test_predicted_sql_identity_uses_manifest_bundle_id(tmp_path):
    (tmp_path / "bundle_manifest.json").write_text(json.dumps({
        "bundle_id": "bundle-123",
        "pipeline_run_id": "run-456",
        "git_commit": "abcdef",
    }), encoding="utf-8")
    result = _predicted_sql_bundle_identity(tmp_path, "fallback", required=True)
    assert "error" not in result
    assert result["candidate_manifest_loaded"] is True
    assert result["bundle_id"] == "bundle-123"
    assert result["pipeline_run_id"] == "run-456"
    assert result["bundle_id_source"] == "bundle_manifest"
    assert result["identity_strength"] == "strong"


def test_manifest_pipeline_run_id_roundtrip():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    d = manifest.to_dict()
    assert d["pipeline_run_id"] == "run-456"
    
    loaded = BundleManifest.from_dict(d)
    assert loaded.pipeline_run_id == "run-456"


def test_manifest_pipeline_run_id_missing_loads_empty():
    d = {
        "bundle_id": "bundle-123",
        "git_commit": "abcdef",
    }
    loaded = BundleManifest.from_dict(d)
    assert loaded.pipeline_run_id == ""


def test_manifest_pipeline_run_id_file_roundtrip(tmp_path):
    manifest = BundleManifest(
        bundle_id="bundle-123",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    manifest_path = tmp_path / "bundle_manifest.json"
    save_manifest(manifest, manifest_path)
    
    loaded = load_manifest(manifest_path)
    assert loaded.pipeline_run_id == "run-456"


def test_pipeline_run_id_alone_fails_identity():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    report = {
        "pipeline_run_id": "run-456",
        # Missing bundle_id and candidate_bundle_dir
    }
    # _verify_report_identity(report, candidate_bundle_dir, manifest, report_name)
    res = _verify_report_identity(report, "some/dir", manifest, "controlled_predicted_sql_report")
    assert res["controlled_predicted_sql_report_pipeline_run_id_only"] is True
    assert res["controlled_predicted_sql_report_identity_stale"] is True
    assert res["controlled_predicted_sql_report_identity_verified"] is False


def test_identity_bundle_id_match():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    report = {
        "bundle_id": "bundle-123",
        "pipeline_run_id": "run-456",
        "commit_sha": "abcdef",
        "generated_at": "2023-01-02T00:00:00Z",
    }
    res = _verify_report_identity(report, "some/dir", manifest, "controlled_predicted_sql_report")
    assert res["controlled_predicted_sql_report_identity_bundle_id_match"] is True
    assert res["controlled_predicted_sql_report_identity_verified"] is True
    assert res["controlled_predicted_sql_report_identity_stale"] is False


def test_identity_candidate_dir_match():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    report = {
        "candidate_bundle_dir": "some/dir",
        "pipeline_run_id": "run-456",
        "commit_sha": "abcdef",
        "generated_at": "2023-01-02T00:00:00Z",
    }
    res = _verify_report_identity(report, "some/dir", manifest, "controlled_predicted_sql_report")
    assert res["controlled_predicted_sql_report_identity_candidate_dir_match"] is True
    assert res["controlled_predicted_sql_report_identity_verified"] is True
    assert res["controlled_predicted_sql_report_identity_stale"] is False


def test_identity_bundle_id_mismatch():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    report = {
        "bundle_id": "bundle-999",
        "pipeline_run_id": "run-456",
    }
    res = _verify_report_identity(report, "some/dir", manifest, "controlled_predicted_sql_report")
    assert res["controlled_predicted_sql_report_identity_bundle_id_match"] is False
    assert res["controlled_predicted_sql_report_identity_verified"] is False
    assert res["controlled_predicted_sql_report_identity_mismatch"] is True


def test_identity_missing_strong_identity():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-01T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    report = {
        "commit_sha": "abcdef",
    }
    res = _verify_report_identity(report, "some/dir", manifest, "controlled_predicted_sql_report")
    assert res["controlled_predicted_sql_report_identity_missing"] is True
    assert res["controlled_predicted_sql_report_identity_verified"] is False


def test_identity_stale_generated_at():
    manifest = BundleManifest(
        bundle_id="bundle-123",
        created_at="2023-01-02T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
    )
    report = {
        "bundle_id": "bundle-123",
        "pipeline_run_id": "run-456",
        "commit_sha": "abcdef",
        "generated_at": "2023-01-01T00:00:00Z",  # generated BEFORE manifest created_at
    }
    res = _verify_report_identity(report, "some/dir", manifest, "controlled_predicted_sql_report")
    assert res["controlled_predicted_sql_report_generated_after_bundle_build"] is False
    assert res["controlled_predicted_sql_report_identity_stale"] is True


def test_model_bundle_validator_rejects_pipeline_run_only_report(tmp_path, monkeypatch):
    import model_bundle.bundle_validator as bundle_validator
    
    # 1. Create a minimal candidate bundle
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = BundleManifest(
        bundle_id="bundle-123",
        status="candidate",
        created_at="2023-01-02T00:00:00Z",
        git_commit="abcdef",
        pipeline_run_id="run-456",
        datasets=["wikisql"],
        paths={
            "retrieval_ir": "retrieval_ir/",
            "evaluation": "evaluation/",
            "generic_training": "generic_training/",
            "configs": "configs/",
        },
        metrics={
            "unsafe_sql_count": 0,
            "sql_validation_rate": 1.0,
            "query_ir_validity_rate": 1.0,
            "unnecessary_join_rate": 0.0,
            "wrong_table_rate": 0.0,
        },
        quality_gate={"passed": True, "required": True, "report_path": "evaluation/model_quality_gate_report.json"},
    )
    save_manifest(manifest, bundle / "bundle_manifest.json")
    
    retrieval = bundle / "retrieval_ir"
    retrieval.mkdir()
    for name in ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl"]:
        (retrieval / name).write_bytes(b"")
    (retrieval / "manifest.json").write_text(
        json.dumps({
            "source_train_file": "train.jsonl",
            "total_examples": 1,
            "by_dataset": {"wikisql": 1},
            "intent_distribution": {"show_records": 1},
            "sql_complexity_distribution": {"simple": 1},
        }),
        encoding="utf-8",
    )
    
    evaluation = bundle / "evaluation"
    evaluation.mkdir()
    (evaluation / "generic_model_evaluation_report.json").write_text(
        json.dumps({"summary": {}, "is_valid_for_quality_gate": True}),
        encoding="utf-8",
    )
    (evaluation / "model_quality_gate_report.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    (evaluation / "classification_metrics_report.json").write_text("{}", encoding="utf-8")
    (evaluation / "calibration_report.json").write_text("{}", encoding="utf-8")
    
    cm_dir = evaluation / "confusion_matrices"
    cm_dir.mkdir()
    for name in ["intent_confusion_matrix.csv", "base_table_confusion_matrix.csv", "join_decision_confusion_matrix.csv", "router_confusion_matrix.csv"]:
        (cm_dir / name).write_text("a,b\n1,0", encoding="utf-8")
        
    generic = bundle / "generic_training"
    generic.mkdir()
    (generic / "dataset_contribution_report.json").write_text(
        json.dumps({
            "datasets_requested": ["wikisql"],
            "leakage_check_passed": True,
            "by_dataset": {"wikisql": {"converted_to_queryir": 1}},
        }),
        encoding="utf-8",
    )
    (generic / "unsupported_sql_report.json").write_text("{}", encoding="utf-8")
    (bundle / "configs").mkdir()

    # 2. Write controlled_predicted_sql_execution_report.json with only pipeline_run_id (no bundle_id or candidate_bundle_dir)
    report = {
        "evaluation_type": "controlled_predicted_sql_execution",
        "measures_model_predictions": True,
        "schema_graph_empty": False,
        "pipeline_run_id": "run-456",
        "predicted_execution_match_rate": 0.9,
        "predicted_execution_success_rate": 0.9,
        "predicted_row_count_match_rate": 0.9,
        "predicted_safe_sql_rate": 1.0,
        "predicted_unsafe_sql_count": 0,
        "unsafe_sql_count": 0,
        "central_sql_validator_used": True,
        "passed": True,
    }
    (evaluation / "controlled_predicted_sql_execution_report.json").write_text(json.dumps(report), encoding="utf-8")

    # Mock the dependencies to avoid heavy imports
    monkeypatch.setattr(bundle_validator, "ROOT", tmp_path)
    monkeypatch.setattr(bundle_validator, "_validate_retrieval_runtime", lambda *_args, **_kwargs: {"passed": True, "calibration_loaded": True})

    # Validate with required mode active
    validator = ModelBundleValidator()
    result = validator.validate(
        bundle,
        config={
            "execution_aware": {
                "controlled_predicted_sql": {
                    "enabled": True,
                    "required_for_full_training": True,
                    "require_report_attached_to_bundle": True,
                }
            }
        }
    )
    
    assert result["passed"] is False
    assert any("pipeline_run_id_only" in issue for issue in result["blocking_issues"])


def _minimal_bundle_with_predicted_report(tmp_path, report):
    bundle = tmp_path / "identity_bundle"
    bundle.mkdir()
    save_manifest(BundleManifest(
        bundle_id="bundle-123",
        status="candidate",
        created_at="2026-01-01T00:00:00Z",
        git_commit="commit-good",
        pipeline_run_id="run-good",
        datasets=["wikisql"],
        paths={
            "retrieval_ir": "retrieval_ir/",
            "evaluation": "evaluation/",
            "generic_training": "generic_training/",
            "configs": "configs/",
        },
        metrics={
            "unsafe_sql_count": 0,
            "sql_validation_rate": 1.0,
            "query_ir_validity_rate": 1.0,
            "unnecessary_join_rate": 0.0,
            "wrong_table_rate": 0.0,
        },
        quality_gate={"passed": True, "required": True},
    ), bundle / "bundle_manifest.json")
    retrieval = bundle / "retrieval_ir"
    retrieval.mkdir()
    for name in ("example_index.pkl", "schema_index.pkl", "pattern_index.pkl"):
        (retrieval / name).write_bytes(b"")
    (retrieval / "manifest.json").write_text(json.dumps({
        "source_train_file": "train.jsonl",
        "total_examples": 1,
        "by_dataset": {"wikisql": 1},
        "intent_distribution": {"show_records": 1},
        "sql_complexity_distribution": {"simple": 1},
    }), encoding="utf-8")
    evaluation = bundle / "evaluation"
    evaluation.mkdir()
    (evaluation / "generic_model_evaluation_report.json").write_text(
        json.dumps({"summary": {}, "is_valid_for_quality_gate": True}), encoding="utf-8",
    )
    (evaluation / "model_quality_gate_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8",
    )
    (evaluation / "classification_metrics_report.json").write_text("{}", encoding="utf-8")
    (evaluation / "calibration_report.json").write_text("{}", encoding="utf-8")
    matrices = evaluation / "confusion_matrices"
    matrices.mkdir()
    for name in (
        "intent_confusion_matrix.csv",
        "base_table_confusion_matrix.csv",
        "join_decision_confusion_matrix.csv",
        "router_confusion_matrix.csv",
    ):
        (matrices / name).write_text("a,b\n1,0", encoding="utf-8")
    (evaluation / "controlled_predicted_sql_execution_report.json").write_text(
        json.dumps(report), encoding="utf-8",
    )
    generic = bundle / "generic_training"
    generic.mkdir()
    (generic / "dataset_contribution_report.json").write_text(json.dumps({
        "datasets_requested": ["wikisql"],
        "leakage_check_passed": True,
        "by_dataset": {"wikisql": {"converted_to_queryir": 1}},
    }), encoding="utf-8")
    (generic / "unsupported_sql_report.json").write_text("{}", encoding="utf-8")
    (bundle / "configs").mkdir()
    return bundle


def _mismatched_predicted_report():
    return {
        "bundle_id": "bundle-123",
        "pipeline_run_id": "run-wrong",
        "commit_sha": "commit-wrong",
        "generated_at": "2026-01-02T00:00:00Z",
        "evaluation_type": "controlled_predicted_sql_execution",
        "measures_model_predictions": True,
        "schema_graph_empty": False,
        "predicted_execution_match_rate": 1.0,
        "predicted_execution_success_rate": 1.0,
        "predicted_row_count_match_rate": 1.0,
        "predicted_safe_sql_rate": 1.0,
        "unsafe_sql_count": 0,
        "central_sql_validator_used": True,
        "passed": True,
    }


def test_public_validator_blocks_commit_and_pipeline_run_mismatch(tmp_path, monkeypatch):
    import model_bundle.bundle_validator as bundle_validator

    bundle = _minimal_bundle_with_predicted_report(tmp_path, _mismatched_predicted_report())
    monkeypatch.setattr(bundle_validator, "ROOT", tmp_path)
    monkeypatch.setattr(
        bundle_validator,
        "_validate_retrieval_runtime",
        lambda *_args, **_kwargs: {"passed": True, "calibration_loaded": True},
    )
    result = ModelBundleValidator().validate(bundle, config={"execution_aware": {
        "controlled_predicted_sql": {
            "enabled": True,
            "required_for_full_training": True,
            "require_report_attached_to_bundle": True,
        },
    }})
    proof = result["lifecycle_proof"]
    assert proof["controlled_predicted_sql_report_commit_matches"] is False
    assert proof["controlled_predicted_sql_report_pipeline_run_matches"] is False
    assert "controlled_predicted_sql_report_identity_stale" in result["blocking_issues"]


def test_public_validator_warns_on_optional_commit_and_pipeline_mismatch(tmp_path, monkeypatch):
    import model_bundle.bundle_validator as bundle_validator

    bundle = _minimal_bundle_with_predicted_report(tmp_path, _mismatched_predicted_report())
    monkeypatch.setattr(bundle_validator, "ROOT", tmp_path)
    monkeypatch.setattr(
        bundle_validator,
        "_validate_retrieval_runtime",
        lambda *_args, **_kwargs: {"passed": True, "calibration_loaded": True},
    )
    result = ModelBundleValidator().validate(bundle, config={"execution_aware": {
        "controlled_predicted_sql": {
            "enabled": True,
            "required_for_full_training": False,
        },
    }})
    proof = result["lifecycle_proof"]
    assert proof["controlled_predicted_sql_report_commit_matches"] is False
    assert proof["controlled_predicted_sql_report_pipeline_run_matches"] is False
    assert "controlled_predicted_sql_report_identity_stale" in result["warnings"]
    assert "controlled_predicted_sql_report_identity_stale" not in result["blocking_issues"]
