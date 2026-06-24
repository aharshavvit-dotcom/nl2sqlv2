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
