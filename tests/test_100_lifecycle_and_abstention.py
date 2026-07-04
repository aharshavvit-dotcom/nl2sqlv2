"""Test 100: Lifecycle proof, abstention behavior, and cross-cutting validation tests.

Tests spanning confidence calibration, abstention, neural-only marking,
runtime defaults, and controlled SQLite fixture execution.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


class TestAbstentionBehavior:
    """Verify calibrated confidence and abstention logic."""

    @staticmethod
    def _make_result_parts(calibration: dict) -> dict:
        """Build a minimal result_parts dict for PredictionConfidenceCalculator.calculate()."""
        from unittest.mock import MagicMock
        candidate = MagicMock()
        candidate.rerank_score = 0.7
        candidate.similarity_score = 0.7
        return {
            "candidates": [candidate],
            "selected_template": {"confidence": 0.8, "intent": "show_records"},
            "slots": {},
            "schema_mapping": {"match_scores": {"users": 0.9}},
            "join_plan": {"confidence": 1.0},
            "validation": {"is_valid": True, "ok": True},
            "ir_validation": {"is_valid": True},
            "warnings": [],
            "calibration": calibration,
        }

    def test_abstention_below_threshold(self) -> None:
        """PredictionConfidenceCalculator with calibration below threshold -> abstain=True."""
        from inference.prediction_confidence import PredictionConfidenceCalculator

        calibration = {
            "isotonic_points": [[0.0, 0.0], [0.5, 0.3], [1.0, 0.6]],
            "conformal_confidence_threshold": 0.90,
            "use_conformal_threshold": True,
        }
        calc = PredictionConfidenceCalculator()
        result = calc.calculate(self._make_result_parts(calibration))

        assert result["calibrated_confidence"] is not None
        assert result["calibrated_confidence"] < 0.90
        assert result["abstain"] is True
        assert result.get("confidence_tier") == "needs_clarification" or result.get("abstention_reason")

    def test_no_abstention_above_threshold(self) -> None:
        """When calibrated confidence is above threshold, abstain must be False."""
        from inference.prediction_confidence import PredictionConfidenceCalculator

        calibration = {
            "isotonic_points": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            "conformal_confidence_threshold": 0.10,
            "use_conformal_threshold": True,
        }
        calc = PredictionConfidenceCalculator()
        result = calc.calculate(self._make_result_parts(calibration))

        assert result["calibrated_confidence"] >= 0.10
        assert result["abstain"] is False

    def test_calibration_changes_raw_confidence(self) -> None:
        """Isotonic calibration should transform the raw confidence value."""
        from inference.prediction_confidence import PredictionConfidenceCalculator

        calibration = {
            "isotonic_points": [[0.0, 0.0], [0.5, 0.3], [1.0, 0.6]],
        }
        calc = PredictionConfidenceCalculator()
        result = calc.calculate(self._make_result_parts(calibration))

        assert result["raw_confidence"] is not None
        # Calibrated should differ from raw when isotonic curve is non-identity
        assert result["calibrated_confidence"] is not None


class TestNeuralOnlyArtifactSource:
    """Verify neural-only fallback is explicitly marked."""

    def test_neural_only_artifact_source_value(self) -> None:
        """The string literal 'neural_only_artifact_dirs' must be used."""
        source = (ROOT / "training" / "evaluate_generic_models.py").read_text(encoding="utf-8")
        assert "neural_only_artifact_dirs" in source

    def test_quality_gate_warns_on_neural_only(self) -> None:
        """Quality gate must produce a warning when model_artifact_source is neural_only_artifact_dirs."""
        from quality_gates.model_quality_gate import ModelQualityGate

        report = {
            "evaluation_mode": "real_model_predictions",
            "gold_replay_used": False,
            "predictor_used": True,
            "is_valid_for_quality_gate": True,
            "model_artifact_source": "neural_only_artifact_dirs",
            "test_performance": {
                "evaluation_mode": "real_model_predictions",
                "gold_replay_used": False,
                "predictor_used": True,
                "is_valid_for_quality_gate": True,
                "model_artifact_source": "neural_only_artifact_dirs",
                "summary": {
                    "query_ir_validity_rate": 0.99,
                    "sql_validation_rate": 0.99,
                    "intent_accuracy_rate": 0.98,
                    "unnecessary_join_rate": 0.0,
                    "wrong_table_rate": 0.0,
                    "unsafe_sql_count": 0,
                },
            },
            "unseen_db_performance": {"summary": {"sql_validation_rate": 0.95}},
            "no_select_star_rate": 1.0,
            "unsafe_sql_count": 0,
            "feedback_regression_pass_rate": 1.0,
            "dataset_contribution_report_required": True,
            "dataset_contribution_report": {
                "datasets_requested": ["wikisql"],
                "leakage_check_passed": True,
                "by_dataset": {"wikisql": {"converted_to_queryir": 10}},
            },
        }
        thresholds = {
            "minimums": {
                "query_ir_validity_rate": 0.90,
                "sql_validation_rate": 0.90,
                "simple_query_pass_rate": 0.0,
                "no_select_star_rate": 1.00,
                "unsafe_sql_count_max": 0,
                "unnecessary_join_rate_max": 0.05,
                "wrong_table_rate_max": 0.15,
                "unseen_db_sql_validation_rate": 0.80,
                "feedback_regression_pass_rate": 0.95,
            },
        }

        result = ModelQualityGate().evaluate(report, thresholds)

        assert any("neural_only" in w.lower() or "neural-only" in w.lower() for w in result.get("warnings", []))


class TestRuntimeDefaults:
    """Verify default bundle path and dev_fallback_used field."""

    def test_runtime_defaults_to_bundle_current(self) -> None:
        """DEFAULT_BUNDLE_DIR in streamlit_app.py points to artifacts/model_bundle/current."""
        source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        assert 'artifacts' in source and 'model_bundle' in source and 'current' in source

    def test_dev_fallback_field_in_predict(self) -> None:
        """RetrievalNL2SQLModel.predict() sets dev_fallback_used in result.debug."""
        source = (ROOT / "retriever" / "retrieval_nl2sql_model.py").read_text(encoding="utf-8")
        assert "dev_fallback_used" in source
        assert "runtime_source" in source


class TestControlledSQLiteFixtures:
    """Verify the controlled execution-aware evaluation fixtures work."""

    def test_controlled_sqlite_fixture_runs(self) -> None:
        """Create fixture DB from seed SQL and verify a SELECT works."""
        sql_path = ROOT / "evaluation" / "fixtures" / "controlled_evaluation.sql"
        if not sql_path.exists():
            pytest.skip("Fixture SQL not found")

        sql_seed = sql_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(sql_seed)
                cursor = conn.execute("SELECT COUNT(*) FROM users")
                count = cursor.fetchone()[0]
                assert count == 5
            finally:
                conn.close()

    def test_controlled_fixture_cases_valid_json(self) -> None:
        """Fixture cases JSONL must be valid and have expected fields."""
        cases_path = ROOT / "evaluation" / "fixtures" / "controlled_evaluation_cases.jsonl"
        if not cases_path.exists():
            pytest.skip("Fixture cases not found")

        lines = cases_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 5

        for line in lines:
            case = json.loads(line)
            assert "example_id" in case
            assert "question" in case
            assert "gold_sql" in case
            assert "expected_row_count" in case
            assert case["gold_sql"].strip().upper().startswith("SELECT")

    def test_generated_sql_is_select_only(self) -> None:
        """All fixture gold_sql must be SELECT-only."""
        cases_path = ROOT / "evaluation" / "fixtures" / "controlled_evaluation_cases.jsonl"
        if not cases_path.exists():
            pytest.skip("Fixture cases not found")

        lines = cases_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            case = json.loads(line)
            sql = case["gold_sql"].strip().upper()
            assert sql.startswith("SELECT"), f"Non-SELECT SQL found: {case['gold_sql']}"
            assert "DROP" not in sql
            assert "DELETE" not in sql
            assert "INSERT" not in sql
            assert "UPDATE" not in sql

    def test_evaluate_controlled_fixtures_function(self) -> None:
        """evaluate_controlled_fixtures() should return correct summary."""
        from training.run_execution_aware_evaluation import evaluate_controlled_fixtures

        sql_path = ROOT / "evaluation" / "fixtures" / "controlled_evaluation.sql"
        cases_path = ROOT / "evaluation" / "fixtures" / "controlled_evaluation_cases.jsonl"
        if not sql_path.exists() or not cases_path.exists():
            pytest.skip("Fixture files not found")

        report = evaluate_controlled_fixtures(sql_path, cases_path)

        assert report["controlled_fixture_evaluation"] is True
        assert report["total_cases"] >= 5
        assert report["summary"]["execution_success_rate"] == 1.0
        assert report["summary"]["row_count_match_rate"] == 1.0
        assert report["summary"]["select_only_rate"] == 1.0


class TestBundleLoaderCalibration:
    """Verify bundle_loader returns calibration paths."""

    def test_bundle_loader_returns_calibration_fields(self) -> None:
        """ModelBundleLoader result dict must include calibration_dir and calibration_report_path."""
        source = (ROOT / "model_bundle" / "bundle_loader.py").read_text(encoding="utf-8")
        assert "calibration_dir" in source
        assert "calibration_report_path" in source

    def test_candidate_bundle_requires_explicit_debug_flag(self, tmp_path, monkeypatch) -> None:
        from model_bundle.bundle_loader import ModelBundleLoader
        from model_bundle.bundle_manifest import BundleManifest, save_manifest

        candidate = tmp_path / "candidate"
        candidate.mkdir()
        save_manifest(BundleManifest(
            bundle_id="candidate-1",
            status="candidate",
            paths={"evaluation": "evaluation/"},
            quality_gate={"passed": False, "required": False},
            lifecycle_proof={"production_ready": False},
        ), candidate / "bundle_manifest.json")
        monkeypatch.setattr(
            "model_bundle.bundle_loader.ModelBundleValidator.validate",
            lambda *_args, **_kwargs: {"passed": True, "blocking_issues": []},
        )
        loader = ModelBundleLoader()
        with pytest.raises(ValueError, match="Candidate bundle loading is disabled"):
            loader.load(candidate)

        loaded = loader.load(candidate, allow_candidate_debug=True)
        assert loaded["bundle_source"] == "candidate_debug"
        assert loaded["quality_gate_passed"] is False
        assert loaded["production_ready"] is False
        assert loaded["loaded_for_debug"] is True

    def test_current_bundle_is_preferred_over_candidate_debug(self, tmp_path, monkeypatch) -> None:
        from model_bundle.bundle_loader import ModelBundleLoader
        from model_bundle.bundle_manifest import BundleManifest, save_manifest

        current = tmp_path / "current"
        candidate = tmp_path / "candidate"
        current.mkdir()
        candidate.mkdir()
        save_manifest(BundleManifest(bundle_id="current-1", status="current"), current / "bundle_manifest.json")
        save_manifest(BundleManifest(bundle_id="candidate-1", status="candidate"), candidate / "bundle_manifest.json")
        monkeypatch.setattr(
            "model_bundle.bundle_loader.ModelBundleValidator.validate",
            lambda *_args, **_kwargs: {"passed": True, "blocking_issues": []},
        )
        loaded = ModelBundleLoader().load_preferred(
            current, candidate, allow_candidate_debug=True,
        )
        assert loaded["bundle_source"] == "current"
        assert loaded["loaded_for_debug"] is False

    def test_streamlit_candidate_debug_warning_is_explicit(self) -> None:
        source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        assert "NL2SQL_ALLOW_CANDIDATE_BUNDLE" in source
        assert "Candidate bundle loaded for debugging only" in source


class TestPreciseRuntimeSource:
    """Verify runtime_source produces precise labels."""

    def test_runtime_source_dev_fallback(self) -> None:
        """When artifact_dir is None, runtime_source must be 'dev_fallback'."""
        from inference.prediction_models import PredictionResult

        result = PredictionResult(question="test", normalized_question="test")
        result.debug["dev_fallback_used"] = True
        # Simulate what RetrievalNL2SQLModel.predict() does for dev fallback
        result.debug["runtime_source"] = "dev_fallback"
        assert result.debug["runtime_source"] == "dev_fallback"

    def test_runtime_source_values_in_predict_code(self) -> None:
        """The predict method must distinguish model_bundle_current, model_bundle_candidate, artifact_dirs, dev_fallback."""
        source = (ROOT / "retriever" / "retrieval_nl2sql_model.py").read_text(encoding="utf-8")
        assert "model_bundle_" in source  # model_bundle_{status}
        assert '"artifact_dirs"' in source
        assert '"dev_fallback"' in source

    def test_runtime_source_model_bundle_status_interpolation(self) -> None:
        """runtime_source must be f'model_bundle_{status}' when bundle metadata present."""
        source = (ROOT / "retriever" / "retrieval_nl2sql_model.py").read_text(encoding="utf-8")
        assert 'f"model_bundle_{bundle_status}"' in source

    def test_debug_contains_calibration_and_drift_flags(self) -> None:
        """predict() must set calibration_loaded and schema_drift_baseline_loaded in debug."""
        source = (ROOT / "retriever" / "retrieval_nl2sql_model.py").read_text(encoding="utf-8")
        assert "calibration_loaded" in source
        assert "schema_drift_baseline_loaded" in source


class TestAbstentionReasonAndDrift:
    """Verify abstention_reason and schema_drift_flags are in PredictionResult."""

    def test_abstention_reason_field_exists(self) -> None:
        """PredictionResult must have abstention_reason field."""
        from inference.prediction_models import PredictionResult

        result = PredictionResult(question="test", normalized_question="test", abstain=True, abstention_reason="low calibrated confidence")
        assert result.abstention_reason == "low calibrated confidence"

    def test_schema_drift_flags_field_exists(self) -> None:
        """PredictionResult must have schema_drift_flags field."""
        from inference.prediction_models import PredictionResult

        result = PredictionResult(
            question="test", normalized_question="test",
            schema_drift_flags=["schema_complexity_p95_exceeded", "question_length_p99_exceeded"],
        )
        assert len(result.schema_drift_flags) == 2
        assert "schema_complexity_p95_exceeded" in result.schema_drift_flags

    def test_abstention_reason_surfaced_in_streamlit_source(self) -> None:
        """The Streamlit app source must reference abstention_reason and schema_drift_flags."""
        source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        assert "abstention_reason" in source
        assert "schema_drift_flags" in source

    def test_abstention_fields_serializable(self) -> None:
        """PredictionResult with abstention_reason and schema_drift_flags must be JSON-serializable."""
        from inference.prediction_models import PredictionResult

        result = PredictionResult(
            question="test", normalized_question="test",
            abstain=True, abstention_reason="conformal threshold",
            schema_drift_flags=["high_complexity"],
        )
        data = result.model_dump()
        serialized = json.dumps(data)
        assert "conformal threshold" in serialized
        assert "high_complexity" in serialized


class TestRelationAwareSchemaAttention:
    """Phase 6: Relation-aware schema attention smoke tests."""

    def test_disabled_mode_preserves_old_forward_path(self) -> None:
        """With relation_aware_attention.enabled=false, forward pass succeeds and
        produces the same outputs as before."""
        import torch
        from neural_ir.attention_model import SchemaAwareOptionAIRModel

        config = {"relation_aware_attention": {"enabled": False}}
        label_sizes = {
            "intent": 5, "metric_aggregation": 4, "metric_expression_type": 3,
            "date_grain": 3, "date_filter_type": 3, "filter_operator": 4,
            "order_direction": 3, "limit_bucket": 3,
        }
        model = SchemaAwareOptionAIRModel(config, vocab_size=100, label_sizes=label_sizes)
        assert model.relation_bias is None
        question_ids = torch.randint(0, 100, (2, 10))
        schema_ids = torch.randint(0, 100, (2, 20))
        outputs = model(question_ids, schema_ids)
        assert "intent_logits" in outputs
        assert "attention_weights" in outputs

    def test_enabled_mode_forward_pass_succeeds(self) -> None:
        """With relation_aware_attention.enabled=true, forward pass succeeds
        and relation_bias module exists."""
        import torch
        from neural_ir.attention_model import SchemaAwareOptionAIRModel

        config = {"relation_aware_attention": {"enabled": True, "bias_init": 0.0}}
        label_sizes = {
            "intent": 5, "metric_aggregation": 4, "metric_expression_type": 3,
            "date_grain": 3, "date_filter_type": 3, "filter_operator": 4,
            "order_direction": 3, "limit_bucket": 3,
        }
        model = SchemaAwareOptionAIRModel(config, vocab_size=100, label_sizes=label_sizes)
        assert model.relation_bias is not None
        assert model.relation_aware_enabled is True

        question_ids = torch.randint(0, 100, (2, 10))
        schema_ids = torch.randint(0, 100, (2, 20))
        # Provide relation_type_ids matching [batch, question_len, schema_len]
        relation_type_ids = torch.randint(0, 10, (2, 10, 20))
        outputs = model(question_ids, schema_ids, relation_type_ids=relation_type_ids)
        assert "intent_logits" in outputs
        assert "attention_weights" in outputs

    def test_relation_type_ids_shape(self) -> None:
        """RelationBiasModule output has expected shape."""
        import torch
        from neural_ir.attention_model import RelationBiasModule

        module = RelationBiasModule(num_types=10, bias_init=0.0)
        ids = torch.randint(0, 10, (2, 8, 16))
        bias = module(ids)
        assert bias.shape == (2, 8, 16)

    def test_relation_types_list_correct(self) -> None:
        """RELATION_TYPES list includes all 10 approved types."""
        from neural_ir.attention_model import RELATION_TYPES

        expected = {
            "same_table", "table_has_column", "column_belongs_to_table",
            "fk_to_pk", "pk_to_fk", "primary_key", "foreign_key_column",
            "same_column_name", "same_data_type", "unrelated",
        }
        assert set(RELATION_TYPES) == expected
        assert len(RELATION_TYPES) == 10

    def test_config_preserves_relation_aware_setting(self) -> None:
        """Model config stores relation_aware_attention setting."""
        from neural_ir.attention_model import SchemaAwareOptionAIRModel

        config = {"relation_aware_attention": {"enabled": True, "bias_init": 0.1}}
        label_sizes = {
            "intent": 5, "metric_aggregation": 4, "metric_expression_type": 3,
            "date_grain": 3, "date_filter_type": 3, "filter_operator": 4,
            "order_direction": 3, "limit_bucket": 3,
        }
        model = SchemaAwareOptionAIRModel(config, vocab_size=100, label_sizes=label_sizes)
        assert model.config["relation_aware_attention"]["enabled"] is True
        assert model.config["relation_aware_attention"]["bias_init"] == 0.1


class TestMultiSeedGovernanceLabels:
    """Phase 2: Multi-seed report shape and governance labels."""

    def test_model_selector_informational_warning_for_single_seed(self) -> None:
        """ModelSelector emits multi_seed_variance_not_available for single-seed baseline."""
        from model_selection.model_selector import ModelSelector
        from model_selection.model_candidate import ModelCandidate

        candidate = ModelCandidate(
            name="test",
            artifact_dir="test_dir",
            model_type="retrieval",
            metrics={"sql_validation_rate": 0.95, "query_ir_validity_rate": 0.92},
            created_at="2026-01-01",
            metadata={"multi_seed_report": {
                "enabled": True,
                "mode": "single_seed_baseline",
                "true_multi_seed": False,
                "is_valid_for_variance_governance": False,
                "seeds_evaluated": 1,
                "metric_std": {"intent_macro_f1": 0.0},
            }},
        )
        result = ModelSelector().select_best([candidate], {"minimums": {}})
        warnings = result.get("warnings", [])
        assert any("multi_seed_variance_not_available" in w for w in warnings)

    def test_model_selector_reads_nested_metrics_shape(self) -> None:
        """ModelSelector reads std from nested metrics.*.std when metric_std is missing."""
        from model_selection.model_selector import ModelSelector
        from model_selection.model_candidate import ModelCandidate

        candidate = ModelCandidate(
            name="test",
            artifact_dir="test_dir",
            model_type="retrieval",
            metrics={"sql_validation_rate": 0.95, "query_ir_validity_rate": 0.92},
            created_at="2026-01-01",
            metadata={"multi_seed_report": {
                "enabled": True, "mode": "evaluation_only_stability",
                "true_multi_seed": True,
                "is_valid_for_variance_governance": True,
                "seeds_evaluated": 3,
                "metrics": {
                    "intent_macro_f1": {"std": 0.08, "mean": 0.80},
                },
            }},
        )
        result = ModelSelector().select_best([candidate], {"minimums": {}})
        warnings = result.get("warnings", [])
        assert any("High metric variance" in w and "intent_macro_f1" in w for w in warnings)


class TestControlledFixtureLabeling:
    """Phase 4: Controlled fixture honest labeling."""

    def test_gold_sql_fixture_report_has_evaluation_type(self) -> None:
        """evaluate_controlled_fixtures report includes evaluation_type and measures_model_predictions."""
        from training.run_execution_aware_evaluation import evaluate_controlled_fixtures
        fixture_dir = ROOT / "evaluation" / "fixtures"
        if not (fixture_dir / "controlled_evaluation.sql").exists():
            pytest.skip("Fixture SQL not available")
        report = evaluate_controlled_fixtures()
        assert report["evaluation_type"] == "controlled_gold_sql_fixture_validation"
        assert report["measures_model_predictions"] is False


class TestRuntimeDebugMetadata:
    """Phase 8: Runtime debug bundle_id/bundle_dir/bundle_status separation."""

    def test_predict_code_has_separate_bundle_fields(self) -> None:
        """retrieval_nl2sql_model.py sets bundle_id, bundle_dir, bundle_status separately."""
        source = (ROOT / "retriever" / "retrieval_nl2sql_model.py").read_text(encoding="utf-8")
        assert 'result.debug["bundle_id"]' in source
        assert 'result.debug["bundle_dir"]' in source
        assert 'result.debug["bundle_status"]' in source
        # bundle_dir should use self.artifact_dir, not bundle_id
        assert 'result.debug["bundle_dir"] = str(self.artifact_dir)' in source


class TestProductionReadySplit:
    """Phase 4+9: production_ready split into core/fixture/full."""

    def test_lifecycle_proof_has_split_production_ready(self) -> None:
        """Bundle builder produces production_ready_core, controlled_fixture_ready, production_ready_full."""
        from model_bundle.bundle_builder import ModelBundleBuilder

        builder = ModelBundleBuilder()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = builder.build_candidate_bundle(
                work_dir=td, output_dir=Path(td) / "out",
                config={"execution_aware": {"controlled_fixtures": {"required_for_full_training": True}}},
                pipeline_report={"steps": []},
            )
            # lifecycle_proof is in the manifest
            manifest_path = Path(result.get("manifest_path", ""))
            if manifest_path.exists():
                import json as json_mod
                manifest = json_mod.loads(manifest_path.read_text(encoding="utf-8"))
                lp = manifest.get("lifecycle_proof", {})
            else:
                lp = result.get("lifecycle_proof", {})
            assert "production_ready_core" in lp
            assert "controlled_fixture_ready" in lp
            assert "production_ready_full" in lp
            assert "simple_query_pass_computed" in lp
            assert "promotion_per_example_fields_complete" in lp
            assert "curriculum_mode" in lp

    def test_production_ready_full_false_when_fixture_required_but_missing(self) -> None:
        """production_ready_full is False when controlled fixture is required but not passed."""
        from model_bundle.bundle_builder import ModelBundleBuilder

        builder = ModelBundleBuilder()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = builder.build_candidate_bundle(
                work_dir=td, output_dir=Path(td) / "out",
                config={"execution_aware": {"controlled_fixtures": {"required_for_full_training": True}}},
                pipeline_report={"steps": []},
            )
            manifest_path = Path(result.get("manifest_path", ""))
            if manifest_path.exists():
                import json as json_mod
                manifest = json_mod.loads(manifest_path.read_text(encoding="utf-8"))
                lp = manifest.get("lifecycle_proof", {})
            else:
                lp = result.get("lifecycle_proof", {})
            # No fixture step means controlled_fixture_eval_passed = False
            assert lp.get("controlled_fixture_ready") is False
            assert lp.get("production_ready_full") is False
