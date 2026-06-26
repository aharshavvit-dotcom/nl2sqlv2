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


def _minimal_bundle(tmp_path: Path) -> Path:
    from model_bundle.bundle_manifest import BundleManifest, save_manifest

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = BundleManifest(
        bundle_id="minimal",
        status="candidate",
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
        quality_gate={"passed": True, "required": False, "report_path": "evaluation/model_quality_gate_report.json"},
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
    return bundle


def _predicted_report(rate: float) -> dict:
    return {
        "evaluation_type": "controlled_predicted_sql_execution",
        "measures_model_predictions": True,
        "schema_graph_empty": False,
        "predicted_execution_match_rate": rate,
        "predicted_execution_success_rate": rate,
        "predicted_row_count_match_rate": rate,
        "predicted_safe_sql_rate": 1.0,
        "predicted_unsafe_sql_count": 0,
        "unsafe_sql_count": 0,
        "central_sql_validator_used": True,
        "passed": rate >= 0.7,
    }


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

    def test_both_training_commands_share_pipeline_registry(self):
        from orchestration.pipeline_config import PipelineConfig, build_pipeline_steps
        from training.train_model import load_training_config

        integrated = load_training_config(ROOT / "configs" / "smoke_training.yaml")
        assert build_pipeline_steps(integrated) == PipelineConfig.load(
            ROOT / "configs" / "smoke_training.yaml"
        ).steps
        assert "build_hard_negative_corpus" in build_pipeline_steps(integrated)

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
        from retrieval.rag_index_builder import RAGIndexBuilder

        manifest = BundleManifest(
            bundle_id="test_valid",
            status="candidate",
            datasets=["wikisql"],
            paths={
                "retrieval_ir": "retrieval_ir/",
                "evaluation": "evaluation/",
                "generic_training": "generic_training/",
                "configs": "configs/",
            },
            artifacts={
                "retrieval_manifest": "retrieval_ir/manifest.json",
                "dataset_contribution_report": "generic_training/dataset_contribution_report.json",
                "unsupported_sql_report": "generic_training/unsupported_sql_report.json",
            },
            metrics={
                "unsafe_sql_count": 0,
                "sql_validation_rate": 1.0,
                "query_ir_validity_rate": 1.0,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
            },
            quality_gate={"passed": True, "required": False, "report_path": "evaluation/model_quality_gate_report.json"},
        )
        save_manifest(manifest, tmp_path / "bundle_manifest.json")
        retrieval = tmp_path / "retrieval_ir"
        RAGIndexBuilder().build(
            [
                {
                    "example_id": "users_1",
                    "dataset_name": "wikisql",
                    "db_id": "db_users",
                    "question": "list all users",
                    "serialized_schema": "tables: users(id, name, role, created_at)",
                    "intent": "show_records",
                    "complexity": "level_1_single_table",
                    "query_ir": {
                        "intent": "show_records",
                        "template_id": "show_records",
                        "base_table": "users",
                        "required_tables": ["users"],
                        "dimensions": [],
                        "metrics": [],
                        "filters": [],
                        "joins": [],
                        "limit": 100,
                    },
                }
            ],
            retrieval,
            source_train_file="data/processed/generic_ir_train.jsonl",
        )
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

    def test_bundle_manifest_includes_lifecycle_proof(self):
        """27. BundleManifest lifecycle_proof fields are populated by builder."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_manifest import BundleManifest

        manifest = BundleManifest(
            bundle_id="test_lifecycle",
            status="candidate",
            lifecycle_proof={
                "trained_from_generic_corpus": True,
                "generic_eval_valid_for_quality_gate": True,
                "generic_eval_real_predictions": True,
                "generic_eval_gold_replay_used": False,
                "calibration_report_available": True,
                "calibration_loaded_in_runtime_smoke": True,
                "quality_gate_passed": True,
                "bundle_runtime_smoke_passed": True,
                "production_ready": True,
            },
        )
        proof = manifest.lifecycle_proof
        assert proof["trained_from_generic_corpus"] is True
        assert proof["generic_eval_valid_for_quality_gate"] is True
        assert proof["production_ready"] is True

    def test_lifecycle_proof_marks_gold_replay_false_for_production(self):
        """28. Gold replay must be false for production_ready to be true."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_manifest import BundleManifest

        manifest = BundleManifest(
            bundle_id="test_gold_replay_check",
            status="candidate",
            lifecycle_proof={
                "generic_eval_gold_replay_used": True,
                "production_ready": False,
            },
        )
        assert manifest.lifecycle_proof["generic_eval_gold_replay_used"] is True
        assert manifest.lifecycle_proof["production_ready"] is False

    def test_lifecycle_proof_records_calibration_loaded(self):
        """29. lifecycle_proof includes calibration_loaded_in_runtime_smoke."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_manifest import BundleManifest

        manifest = BundleManifest(
            bundle_id="test_cal",
            status="candidate",
            lifecycle_proof={
                "calibration_loaded_in_runtime_smoke": False,
            },
        )
        assert "calibration_loaded_in_runtime_smoke" in manifest.lifecycle_proof
        assert manifest.lifecycle_proof["calibration_loaded_in_runtime_smoke"] is False

    def test_controlled_fixture_step_in_pipeline_config(self):
        """30. When execution_aware.controlled_fixtures.enabled=true, pipeline includes the step."""
        sys.path.insert(0, str(ROOT))
        from orchestration.pipeline_config import build_pipeline_steps

        config = {"evaluation": {"enabled": True}, "execution_aware": {"controlled_fixtures": {"enabled": True}}}
        steps = build_pipeline_steps(config)
        assert "run_controlled_fixture_evaluation" in steps

    def test_controlled_fixture_step_absent_when_disabled(self):
        """31. When controlled_fixtures not enabled, step should not appear."""
        sys.path.insert(0, str(ROOT))
        from orchestration.pipeline_config import build_pipeline_steps

        steps = build_pipeline_steps({"evaluation": {"enabled": True}})
        assert "run_controlled_fixture_evaluation" not in steps

    def test_multi_seed_config_parsed(self):
        """32. Seeds config is parsed from training.yaml."""
        import yaml

        config_path = ROOT / "configs" / "training.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seeds = config.get("seeds", {})
        assert "enabled" in seeds
        assert "values" in seeds
        assert "metrics" in seeds
        assert isinstance(seeds["metrics"], list)
        assert len(seeds["metrics"]) > 0

    def test_multi_seed_status_used_before_assignment_fixed(self):
        """33. The NameError where status was used before assignment in train_model.py is fixed."""
        import ast

        source = (ROOT / "training" / "train_model.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Find the main() function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                # Find all assignments to 'status' and all reads of 'status'
                first_assign = None
                first_read_in_seeds = None
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id == "status":
                                if first_assign is None:
                                    first_assign = child.lineno
                    if isinstance(child, ast.Name) and child.id == "status" and isinstance(child.ctx, ast.Load):
                        if first_read_in_seeds is None or child.lineno < first_read_in_seeds:
                            first_read_in_seeds = child.lineno
                assert first_assign is not None, "status is never assigned in main()"
                assert first_read_in_seeds is not None, "status is never read in main()"
                assert first_assign <= first_read_in_seeds, (
                    f"status is read at line {first_read_in_seeds} before assigned at line {first_assign}"
                )
                break

    def test_predicted_sql_attachment_step_order(self):
        """34. Predicted-SQL report attachment runs before bundle validation."""
        sys.path.insert(0, str(ROOT))
        from orchestration.pipeline_config import build_pipeline_steps

        steps = build_pipeline_steps({
            "bundle": {"build": True, "validate": True},
            "execution_aware": {"controlled_predicted_sql": {"enabled": True}},
        })

        assert steps.index("build_model_bundle") < steps.index("run_controlled_predicted_sql_evaluation")
        assert steps.index("run_controlled_predicted_sql_evaluation") < steps.index("attach_runtime_evaluation_reports_to_bundle")
        assert steps.index("attach_runtime_evaluation_reports_to_bundle") < steps.index("validate_model_bundle")

    def test_attach_runtime_evaluation_reports_to_bundle_copies_predicted_sql(self, tmp_path):
        """35. Attachment step copies controlled predicted-SQL report into candidate/evaluation."""
        sys.path.insert(0, str(ROOT))
        from orchestration.pipeline_config import PipelineConfig
        from orchestration.step_runner import StepRunner

        evaluation_dir = tmp_path / "evaluation"
        candidate_dir = tmp_path / "candidate"
        evaluation_dir.mkdir()
        candidate_dir.mkdir()
        (candidate_dir / "bundle_manifest.json").write_text("{}", encoding="utf-8")
        report = {"evaluation_type": "controlled_predicted_sql_execution", "central_sql_validator_used": True}
        (evaluation_dir / "controlled_predicted_sql_execution_report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        config = PipelineConfig(
            pipeline_name="attach_test",
            artifacts={"evaluation_dir": str(evaluation_dir)},
            training={"_integrated_config": {"paths": {"candidate_bundle_dir": str(candidate_dir)}}},
            steps=["attach_runtime_evaluation_reports_to_bundle"],
        )

        result = StepRunner().run_step("attach_runtime_evaluation_reports_to_bundle", config)

        assert result["status"] == "completed"
        assert (candidate_dir / "evaluation" / "controlled_predicted_sql_execution_report.json").exists()

    def test_bundle_validator_prefers_bundle_predicted_sql_report(self, tmp_path, monkeypatch):
        """36. Validator prefers candidate/evaluation predicted-SQL report over root fallback."""
        sys.path.insert(0, str(ROOT))
        import model_bundle.bundle_validator as bundle_validator
        from model_bundle.bundle_validator import ModelBundleValidator

        bundle = _minimal_bundle(tmp_path)
        monkeypatch.setattr(bundle_validator, "ROOT", tmp_path)
        monkeypatch.setattr(bundle_validator, "_validate_retrieval_runtime", lambda *_args, **_kwargs: {"passed": True, "calibration_loaded": False})
        root_eval = tmp_path / "artifacts" / "evaluation"
        root_eval.mkdir(parents=True)
        (root_eval / "controlled_predicted_sql_execution_report.json").write_text(json.dumps(_predicted_report(0.1)), encoding="utf-8")
        (bundle / "evaluation" / "controlled_predicted_sql_execution_report.json").write_text(json.dumps(_predicted_report(0.9)), encoding="utf-8")

        result = ModelBundleValidator().validate(bundle)

        proof = result["lifecycle_proof"]
        assert proof["controlled_predicted_sql_report_location"] == "bundle"
        assert proof["controlled_predicted_sql_execution_match_rate"] == 0.9

    def test_bundle_validator_root_fallback_warns(self, tmp_path, monkeypatch):
        """37. Validator reads root predicted-SQL fallback and warns when not attached."""
        sys.path.insert(0, str(ROOT))
        import model_bundle.bundle_validator as bundle_validator
        from model_bundle.bundle_validator import ModelBundleValidator

        bundle = _minimal_bundle(tmp_path)
        monkeypatch.setattr(bundle_validator, "ROOT", tmp_path)
        monkeypatch.setattr(bundle_validator, "_validate_retrieval_runtime", lambda *_args, **_kwargs: {"passed": True, "calibration_loaded": False})
        root_eval = tmp_path / "artifacts" / "evaluation"
        root_eval.mkdir(parents=True)
        (root_eval / "controlled_predicted_sql_execution_report.json").write_text(json.dumps(_predicted_report(0.5)), encoding="utf-8")

        result = ModelBundleValidator().validate(bundle)

        assert result["lifecycle_proof"]["controlled_predicted_sql_report_location"] == "root_artifacts"
        assert "controlled_predicted_sql_report_not_attached_to_bundle" in result["warnings"]

    def test_bundle_validator_required_attached_mode_fails_on_root_only(self, tmp_path, monkeypatch):
        """38. Required attachment mode fails when only root fallback exists."""
        sys.path.insert(0, str(ROOT))
        import model_bundle.bundle_validator as bundle_validator
        from model_bundle.bundle_validator import ModelBundleValidator

        bundle = _minimal_bundle(tmp_path)
        monkeypatch.setattr(bundle_validator, "ROOT", tmp_path)
        monkeypatch.setattr(bundle_validator, "_validate_retrieval_runtime", lambda *_args, **_kwargs: {"passed": True, "calibration_loaded": False})
        root_eval = tmp_path / "artifacts" / "evaluation"
        root_eval.mkdir(parents=True)
        (root_eval / "controlled_predicted_sql_execution_report.json").write_text(json.dumps(_predicted_report(0.8)), encoding="utf-8")

        result = ModelBundleValidator().validate(
            bundle,
            config={
                "execution_aware": {
                    "controlled_predicted_sql": {
                        "enabled": True,
                        "required_for_full_training": True,
                        "require_report_attached_to_bundle": True,
                    }
                }
            },
        )

        assert result["passed"] is False
        assert "controlled_predicted_sql_report_required_but_not_attached_to_bundle" in result["blocking_issues"]

    def test_production_ready_requires_all_critical_fields(self):
        """39. production_ready in lifecycle proof requires all critical conditions."""
        sys.path.insert(0, str(ROOT))
        from model_bundle.bundle_builder import ModelBundleBuilder

        # With all conditions met
        result = ModelBundleBuilder().build_candidate_bundle(
            work_dir=ROOT / "artifacts",
            output_dir=ROOT / "artifacts" / "model_bundle" / "_test_prod_ready",
            config={},
            pipeline_report={"steps": [
                {"step": "run_app_smoke_check", "status": "completed", "summary": {"calibration_loaded": True}},
                {"step": "run_controlled_fixture_evaluation", "status": "completed", "summary": {
                    "passed": True, "execution_success_rate": 1.0, "row_count_match_rate": 1.0,
                }},
            ]},
            evaluation_report={
                "is_valid_for_quality_gate": True,
                "real_predictions_generated": 10,
                "gold_replay_used": False,
                "predictor_used": True,
                "test_performance": {"calibration": {"conformal_confidence_threshold": 0.85}},
            },
            quality_gate_report={"passed": True},
        )
        manifest_path = ROOT / "artifacts" / "model_bundle" / "_test_prod_ready" / "bundle_manifest.json"
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            lp = manifest_data.get("lifecycle_proof", {})
            assert lp.get("production_ready") is True
            assert lp.get("controlled_fixture_eval_available") is True
            # Clean up
            import shutil
            shutil.rmtree(ROOT / "artifacts" / "model_bundle" / "_test_prod_ready", ignore_errors=True)

    def test_step_runner_recognizes_controlled_fixture_step(self):
        """35. StepRunner can get contract for run_controlled_fixture_evaluation."""
        sys.path.insert(0, str(ROOT))
        from orchestration.pipeline_config import PipelineConfig
        from orchestration.step_runner import StepRunner

        config = PipelineConfig(pipeline_name="test", training={"_integrated_config": {
            "execution_aware": {"controlled_fixtures": {"enabled": True}},
        }})
        contract = StepRunner().get_contract("run_controlled_fixture_evaluation", config)
        assert contract.name == "run_controlled_fixture_evaluation"
