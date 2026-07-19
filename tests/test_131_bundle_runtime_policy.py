"""Unit tests for ModelBundleLoader and bundle runtime routing policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from unittest.mock import patch
from model_bundle.bundle_manifest import BundleManifest, save_manifest
from model_bundle.bundle_loader import ModelBundleLoader
from model_bundle.bundle_validator import ModelBundleValidator
from inference.prediction_orchestrator import PredictionOrchestrator


@pytest.fixture
def mock_current_bundle(tmp_path):
    bundle = tmp_path / "model_bundle" / "current"
    bundle.mkdir(parents=True)
    
    manifest = BundleManifest(
        bundle_id="prod_active",
        status="current",
        datasets=["wikisql"],
        paths={
            "retrieval_ir": "retrieval_ir/",
            "evaluation": "evaluation/",
            "generic_training": "generic_training/",
            "configs": "configs/",
        },
        metrics={
            "unsafe_sql_count": 0,
            "sql_validation_rate": 0.95,
            "query_ir_validity_rate": 0.95,
        },
        lifecycle_proof={"production_ready": True},
        quality_gate={"passed": True, "required": True},
        quality_gate_passed=True,
        dataset_contribution_status={"passed": True},
        sklearn_artifact_version={"sklearn_version": "1.2.2"},
        eligible_for_promotion=True,
        production_ready_full=True,
        quality_gate_mode="production",
        routing_policy={
            "neural_fallback_enabled": True,
            "neural_trigger_threshold": 0.74,
        }
    )
    save_manifest(manifest, bundle / "bundle_manifest.json")
    
    (bundle / "retrieval_ir").mkdir()
    (bundle / "retrieval_ir" / "manifest.json").write_text(json.dumps({"sklearn_artifact_metadata": {"sklearn_version": "1.2.2"}}))
    (bundle / "evaluation").mkdir()
    (bundle / "evaluation" / "model_quality_gate_report.json").write_text('{"passed": true}')
    (bundle / "evaluation" / "generic_model_evaluation_report.json").write_text('{}')
    (bundle / "generic_training").mkdir()
    (bundle / "generic_training" / "dataset_contribution_report.json").write_text('{"full_training_dataset_minimums_passed": true}')
    (bundle / "generic_training" / "unsupported_sql_report.json").write_text('{}')
    (bundle / "configs").mkdir()
    
    # Write a dummy model.pt to allow loading neural model
    (bundle / "neural_ir").mkdir()
    (bundle / "neural_ir" / "model.pt").write_text("fake weights")
    (bundle / "neural_ir" / "hybrid_calibration.json").write_text('{"retrieval_ir_high_confidence_threshold": 0.8}')
    (bundle / "neural_ir" / "confidence_calibration.json").write_text('{}')
    
    return bundle


def test_production_loader_enforces_current_only(mock_current_bundle):
    # Set manifest back to candidate to check rejection
    manifest_path = mock_current_bundle / "bundle_manifest.json"
    manifest = BundleManifest.from_dict(json.loads(manifest_path.read_text()))
    manifest.status = "candidate"
    save_manifest(manifest, manifest_path)
    
    with patch("model_bundle.bundle_validator.ModelBundleValidator.validate", return_value={"passed": True}):
        # Trying to load candidate in production must fail
        with pytest.raises(ValueError, match="Production mode forbids candidate bundle loading"):
            ModelBundleLoader.load_current(mock_current_bundle, runtime_mode="production")


def test_production_loader_succeeds_on_current(mock_current_bundle):
    with patch("model_bundle.bundle_validator.ModelBundleValidator.validate", return_value={"passed": True}):
        bundle_info = ModelBundleLoader.load_current(mock_current_bundle, runtime_mode="production")
    assert bundle_info["bundle_source"] == "current"
    assert bundle_info["loaded_for_debug"] is False


def test_prediction_orchestrator_loads_routing_thresholds_from_bundle(mock_current_bundle):
    # Setup production env
    os.environ["NL2SQL_ENV"] = "production"
    try:
        with patch("model_bundle.bundle_validator.ModelBundleValidator.validate", return_value={"passed": True}):
            bundle_info = ModelBundleLoader.load_current(mock_current_bundle, runtime_mode="production")
        
        # Instantiate PredictionOrchestrator passing the loaded bundle
        orchestrator = PredictionOrchestrator(bundle=bundle_info)
        
        # Verify routing policy loaded from bundle manifest
        assert orchestrator.use_neural_ir_fallback is True
        assert orchestrator.neural_ir_threshold == 0.74
        assert orchestrator.bundle_identity == "prod_active"
    finally:
        os.environ["NL2SQL_ENV"] = "development"


def test_bundle_validator_requires_full_retrain_seed_evidence(mock_current_bundle):
    result = ModelBundleValidator().validate(
        mock_current_bundle,
        config={
            "quality_gate": {"mode": "production"},
            "seeds": {"require_full_retrain_for_production": True},
        },
    )

    assert "multi_seed_full_retrain_required_but_missing_or_invalid" in result["blocking_issues"]
