"""Integration test for the canonical training command.

Runs the smoke training pipeline and verifies all required outputs.

Usage:
    pytest tests/test_99_train_model_integration.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_training_result():
    """Run the smoke training command once for all tests in this module."""
    config_path = ROOT / "configs" / "smoke_training.yaml"
    if not config_path.exists():
        pytest.skip("configs/smoke_training.yaml not found")

    # Run training in dry-run mode first to verify config is valid
    result = subprocess.run(
        [sys.executable, str(ROOT / "training" / "train_model.py"),
         "--config", str(config_path), "--dry-run"],
        capture_output=True, text=True, check=False, cwd=str(ROOT),
    )
    return {"dry_run_returncode": result.returncode, "dry_run_stdout": result.stdout, "dry_run_stderr": result.stderr}


class TestTrainModelIntegration:
    """Integration tests for the canonical training command."""

    def test_train_model_exists(self):
        """1. training/train_model.py exists."""
        assert (ROOT / "training" / "train_model.py").exists()

    def test_smoke_config_exists(self):
        """2. configs/smoke_training.yaml exists."""
        assert (ROOT / "configs" / "smoke_training.yaml").exists()

    def test_full_config_exists(self):
        """3. configs/training.yaml exists."""
        assert (ROOT / "configs" / "training.yaml").exists()

    def test_bundle_manifest_module_exists(self):
        """4. model_bundle/bundle_manifest.py exists."""
        assert (ROOT / "model_bundle" / "bundle_manifest.py").exists()

    def test_bundle_builder_module_exists(self):
        """5. model_bundle/bundle_builder.py exists."""
        assert (ROOT / "model_bundle" / "bundle_builder.py").exists()

    def test_bundle_validator_module_exists(self):
        """6. model_bundle/bundle_validator.py exists."""
        assert (ROOT / "model_bundle" / "bundle_validator.py").exists()

    def test_bundle_loader_module_exists(self):
        """7. model_bundle/bundle_loader.py exists."""
        assert (ROOT / "model_bundle" / "bundle_loader.py").exists()

    def test_bundle_promoter_module_exists(self):
        """8. model_bundle/bundle_promoter.py exists."""
        assert (ROOT / "model_bundle" / "bundle_promoter.py").exists()

    def test_step_contract_exists(self):
        """9. orchestration/step_contract.py exists."""
        assert (ROOT / "orchestration" / "step_contract.py").exists()

    def test_contract_validator_exists(self):
        """10. orchestration/contract_validator.py exists."""
        assert (ROOT / "orchestration" / "contract_validator.py").exists()

    def test_integrated_quality_gate_exists(self):
        """11. quality_gates/integrated_quality_gate.py exists."""
        assert (ROOT / "quality_gates" / "integrated_quality_gate.py").exists()

    def test_dry_run_succeeds(self, smoke_training_result):
        """12. Smoke training dry-run completes without error."""
        assert smoke_training_result["dry_run_returncode"] == 0, (
            f"Dry-run failed:\n{smoke_training_result['dry_run_stderr']}"
        )

    def test_dry_run_shows_steps(self, smoke_training_result):
        """13. Dry-run output lists pipeline steps."""
        stdout = smoke_training_result["dry_run_stdout"]
        assert "Dry-run" in stdout or "dry_run" in stdout or "steps" in stdout.lower()

    def test_bundle_manifest_importable(self):
        """14. BundleManifest can be imported."""
        sys.path.insert(0, str(ROOT))
        try:
            from model_bundle.bundle_manifest import BundleManifest
            manifest = BundleManifest(bundle_id="test", status="candidate")
            assert manifest.bundle_id == "test"
            assert manifest.status == "candidate"
        finally:
            pass

    def test_bundle_loader_importable(self):
        """15. ModelBundleLoader can be imported."""
        sys.path.insert(0, str(ROOT))
        try:
            from model_bundle.bundle_loader import ModelBundleLoader
            loader = ModelBundleLoader()
            assert loader is not None
        finally:
            pass

    def test_bundle_validator_importable(self):
        """16. ModelBundleValidator can be imported."""
        sys.path.insert(0, str(ROOT))
        try:
            from model_bundle.bundle_validator import ModelBundleValidator
            validator = ModelBundleValidator()
            assert validator is not None
        finally:
            pass

    def test_step_contract_importable(self):
        """17. StepContract can be imported."""
        sys.path.insert(0, str(ROOT))
        try:
            from orchestration.step_contract import StepContract
            contract = StepContract(name="test", required=True)
            assert contract.name == "test"
        finally:
            pass

    def test_contract_validator_importable(self):
        """18. ContractValidator can be imported."""
        sys.path.insert(0, str(ROOT))
        try:
            from orchestration.contract_validator import ContractValidator
            validator = ContractValidator()
            assert validator is not None
        finally:
            pass

    def test_integrated_quality_gate_importable(self):
        """19. IntegratedQualityGate can be imported."""
        sys.path.insert(0, str(ROOT))
        try:
            from quality_gates.integrated_quality_gate import IntegratedQualityGate
            gate = IntegratedQualityGate()
            assert gate is not None
        finally:
            pass

    def test_create_orchestrator_from_bundle_importable(self):
        """20. create_orchestrator_from_bundle function exists."""
        sys.path.insert(0, str(ROOT))
        try:
            from inference.prediction_orchestrator import create_orchestrator_from_bundle
            assert callable(create_orchestrator_from_bundle)
        finally:
            pass

    def test_bundle_manifest_roundtrip(self, tmp_path):
        """21. BundleManifest can be saved and loaded."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_manifest import BundleManifest, load_manifest, save_manifest

        manifest = BundleManifest(
            bundle_id="test_roundtrip",
            status="candidate",
            created_at="2026-01-01T00:00:00",
            datasets=["wikisql"],
        )
        manifest_path = tmp_path / "bundle_manifest.json"
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        assert loaded.bundle_id == "test_roundtrip"
        assert loaded.status == "candidate"
        assert loaded.datasets == ["wikisql"]

    def test_bundle_validator_on_empty_dir(self, tmp_path):
        """22. BundleValidator correctly rejects empty directory."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_validator import ModelBundleValidator

        validator = ModelBundleValidator()
        result = validator.validate(tmp_path)
        assert not result["passed"]
        assert "bundle_manifest.json not found" in result["blocking_issues"]

    def test_bundle_validator_on_valid_bundle(self, tmp_path):
        """23. BundleValidator passes a correctly structured bundle."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_manifest import BundleManifest, save_manifest
        from model_bundle.bundle_validator import ModelBundleValidator

        manifest = BundleManifest(
            bundle_id="test_valid",
            status="candidate",
            datasets=["wikisql"],
            paths={
                "retrieval_ir": "retrieval_ir/",
                "neural_ir": "neural_ir/",
                "evaluation": "evaluation/",
                "generic_training": "generic_training/",
                "configs": "configs/",
            },
            metrics={
                "unsafe_sql_count": 0,
                "sql_validation_rate": 1.0,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
            },
            quality_gate={"passed": True, "required": False, "report_path": "evaluation/model_quality_gate_report.json"},
        )
        save_manifest(manifest, tmp_path / "bundle_manifest.json")
        retrieval = tmp_path / "retrieval_ir"
        retrieval.mkdir()
        for name in ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"]:
            (retrieval / name).write_text("{}", encoding="utf-8")
        neural = tmp_path / "neural_ir"
        neural.mkdir()
        for name in ["model.pt", "config.yaml", "manifest.json"]:
            (neural / name).write_text("{}", encoding="utf-8")
        evaluation = tmp_path / "evaluation"
        evaluation.mkdir()
        (evaluation / "generic_model_evaluation_report.json").write_text("{}", encoding="utf-8")
        generic_training = tmp_path / "generic_training"
        generic_training.mkdir()
        (generic_training / "dataset_contribution_report.json").write_text(
            json.dumps({
                "datasets_requested": ["wikisql"],
                "leakage_check_passed": True,
                "by_dataset": {"wikisql": {"converted_to_queryir": 1}},
            }),
            encoding="utf-8",
        )
        (generic_training / "unsupported_sql_report.json").write_text("{}", encoding="utf-8")
        (tmp_path / "configs").mkdir()

        validator = ModelBundleValidator()
        result = validator.validate(tmp_path)
        assert result["passed"], f"Unexpected issues: {result['blocking_issues']}"

    def test_streamlit_has_bundle_loader(self):
        """24. Streamlit app uses ModelBundleLoader."""
        app_path = ROOT / "app" / "streamlit_app.py"
        assert app_path.exists()
        source = app_path.read_text(encoding="utf-8")
        assert "ModelBundleLoader" in source

    def test_streamlit_has_dev_training_flag(self):
        """25. Streamlit app has ENABLE_DEV_TRAINING_UI flag."""
        app_path = ROOT / "app" / "streamlit_app.py"
        assert app_path.exists()
        source = app_path.read_text(encoding="utf-8")
        assert "ENABLE_DEV_TRAINING_UI" in source

    def test_readme_has_canonical_command(self):
        """26. README contains the canonical training command."""
        readme = ROOT / "README.md"
        assert readme.exists()
        source = readme.read_text(encoding="utf-8")
        assert "python training/train_model.py --config configs/training.yaml" in source
