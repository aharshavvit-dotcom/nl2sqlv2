"""Tests for quality gate semantic checks.

Validates:
- Safe-but-wrong SQL blocks production promotion
- Low filter accuracy blocks production
- Low projection accuracy blocks production
- Artifact-dir evaluation cannot promote
- Baseline/debug modes cannot promote
- Full bundle production can promote only when all checks pass
- Semantic grounding readiness is tracked
"""

from __future__ import annotations

import pytest
from quality_gates.model_quality_gate import ModelQualityGate


def _base_report(
    *,
    mode: str = "production",
    sql_validation_rate: float = 1.0,
    simple_query_pass_rate: float = 0.96,
    query_ir_validity_rate: float = 0.98,
    unsafe_sql_count: int = 0,
    unnecessary_join_rate: float = 0.01,
    wrong_table_rate: float = 0.02,
    evaluation_mode: str = "real_model_predictions",
    predictor_used: bool = True,
    rows_evaluated: int = 292,
    real_predictions_generated: int = 292,
    is_valid_for_quality_gate: bool = True,
    semantic_evaluation: dict | None = None,
) -> dict:
    """Build a minimal evaluation report for gate testing."""
    return {
        "quality_gate_mode": mode,
        "evaluation_mode": evaluation_mode,
        "predictor_used": predictor_used,
        "rows_evaluated": rows_evaluated,
        "real_predictions_generated": real_predictions_generated,
        "is_valid_for_quality_gate": is_valid_for_quality_gate,
        "test_performance": {
            "summary": {
                "sql_validation_rate": sql_validation_rate,
                "simple_query_pass_rate": simple_query_pass_rate,
                "query_ir_validity_rate": query_ir_validity_rate,
                "unsafe_sql_count": unsafe_sql_count,
                "unnecessary_join_rate": unnecessary_join_rate,
                "wrong_table_rate": wrong_table_rate,
                "no_select_star_rate": 1.0,
                "intent_accuracy_rate": 0.92,
                "intent_macro_f1": 0.85,
                "base_table_accuracy_rate": 0.95,
                "base_table_macro_f1": 0.90,
                "join_decision_macro_f1": 0.88,
                "router_accuracy": 0.90,
                "router_macro_f1": 0.85,
                "gold_comparison_score": 0.80,
                "structural_sql_match_rate": 0.75,
                "execution_match_rate": 0.72,
                "filter_column_accuracy_rate": 0.75,
                "filter_value_accuracy_rate": 0.72,
                "dimension_column_accuracy_rate": 0.68,
                "projection_exact_match_rate": 0.73,
                "semantic_evaluation": semantic_evaluation or _default_semantic_eval(),
            },
            "classification_metrics": {
                "intent": {"accuracy": 0.92, "macro_f1": 0.85},
                "base_table": {"accuracy": 0.95, "macro_f1": 0.90},
                "join_decision": {"macro_f1": 0.88},
                "router": {"accuracy": 0.90, "macro_f1": 0.85},
            },
            "calibration": {
                "sample_count": 292,
                "expected_calibration_error": 0.05,
                "brier_score": 0.15,
                "calibration_degenerate": False,
                "confidence_unique_value_count": 50,
                "confidence_std": 0.15,
                "confidence_bucket_coverage_count": 8,
            },
        },
        "summary": {
            "unsafe_sql_count": unsafe_sql_count,
        },
    }


def _default_semantic_eval():
    return {
        "simple_query_semantic_pass_rate": 0.65,
        "simple_query_safety_pass_rate": 1.0,
        "simple_query_validity_pass_rate": 1.0,
        "simple_query_table_pass_rate": 0.95,
        "simple_query_projection_pass_rate": 0.73,
        "simple_query_filter_pass_rate": 0.72,
        "projection_exact_match": {"value": 0.73, "numerator": 73, "denominator": 100, "applicable_cases": 100, "excluded_cases": 192},
        "filter_column_accuracy": {"value": 0.75, "numerator": 75, "denominator": 100, "applicable_cases": 100, "excluded_cases": 192},
        "filter_value_accuracy": {"value": 0.72, "numerator": 72, "denominator": 100, "applicable_cases": 100, "excluded_cases": 192},
        "dimension_column_accuracy": {"value": 0.68, "numerator": 68, "denominator": 100, "applicable_cases": 100, "excluded_cases": 192},
    }


def _thresholds(*, semantic: dict | None = None):
    return {
        "minimums": {
            "query_ir_validity_rate": 0.90,
            "sql_validation_rate": 0.90,
            "simple_query_pass_rate": 0.80,
            "simple_query_pass_rate_production": 0.95,
            "no_select_star_rate": 1.00,
            "unsafe_sql_count_max": 0,
            "unnecessary_join_rate_max": 0.05,
            "wrong_table_rate_max": 0.15,
            "unseen_db_sql_validation_rate": 0.80,
            "gold_comparison_score_min": 0.75,
            "intent_macro_f1_min": 0.80,
            "base_table_accuracy_min": 0.85,
            "base_table_macro_f1_min": 0.80,
            "join_decision_macro_f1_min": 0.85,
            "router_accuracy_min": 0.85,
            "router_macro_f1_min": 0.80,
        },
        "semantic": semantic or {
            "minimum_applicable_cases": 50,
            "projection_exact_match_rate": {"production_min": 0.70},
            "filter_column_accuracy_rate": {"production_min": 0.70},
            "filter_value_accuracy_rate": {"production_min": 0.70},
            "dimension_column_accuracy_rate": {"production_min": 0.65},
            "safe_but_wrong_sql_rate_max": {"production_max": 0.30},
        },
    }


class TestSafeButWrongBlocksProduction:
    def test_high_safe_but_wrong_blocks(self):
        report = _base_report()
        report["controlled_predicted_sql_execution"] = {
            "safe_but_wrong_sql_rate": 0.50,
            "predicted_execution_match_rate": 0.72,
            "predicted_safe_sql_rate": 1.0,
        }
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        failed_metrics = [c["metric"] for c in result["failed_checks"]]
        assert "controlled_predicted_sql_safe_but_wrong_sql_rate" in failed_metrics

    def test_low_safe_but_wrong_passes(self):
        report = _base_report()
        report["controlled_predicted_sql_execution"] = {
            "safe_but_wrong_sql_rate": 0.20,
            "predicted_execution_match_rate": 0.72,
            "predicted_safe_sql_rate": 1.0,
        }
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        failed_metrics = [c["metric"] for c in result["failed_checks"]]
        assert "controlled_predicted_sql_safe_but_wrong_sql_rate" not in failed_metrics


class TestLowFilterAccuracyBlocksProduction:
    def test_low_filter_accuracy_blocks(self):
        sem = _default_semantic_eval()
        sem["filter_column_accuracy"]["value"] = 0.40
        report = _base_report(semantic_evaluation=sem)
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        failed_metrics = [c["metric"] for c in result["failed_checks"]]
        assert "filter_column_accuracy_rate" in failed_metrics

    def test_good_filter_accuracy_passes(self):
        report = _base_report()
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        failed_metrics = [c["metric"] for c in result["failed_checks"]]
        assert "filter_column_accuracy_rate" not in failed_metrics


class TestLowProjectionAccuracyBlocksProduction:
    def test_low_projection_blocks(self):
        sem = _default_semantic_eval()
        sem["projection_exact_match"]["value"] = 0.40
        report = _base_report(semantic_evaluation=sem)
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        failed_metrics = [c["metric"] for c in result["failed_checks"]]
        assert "projection_exact_match_rate" in failed_metrics


class TestArtifactDirEvalCannotPromote:
    def test_artifact_dir_blocks_gate(self):
        report = _base_report(is_valid_for_quality_gate=False)
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        assert result["eligible_for_promotion"] is False


class TestBaselineCannotPromote:
    def test_baseline_mode_not_eligible(self):
        report = _base_report(mode="baseline")
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        assert result["eligible_for_promotion"] is False

    def test_debug_mode_not_eligible(self):
        report = _base_report(mode="debug")
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        assert result["eligible_for_promotion"] is False


class TestFullBundleProductionPromote:
    def test_all_pass_can_promote(self):
        report = _base_report()
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        # Should have no semantic blocking failures since defaults pass thresholds
        semantic_failures = [
            c for c in result["failed_checks"]
            if c["metric"] in {
                "projection_exact_match_rate",
                "filter_column_accuracy_rate",
                "filter_value_accuracy_rate",
                "dimension_column_accuracy_rate",
            }
        ]
        assert semantic_failures == []

    def test_semantic_grounding_ready_when_all_above_threshold(self):
        report = _base_report()
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        readiness = result["production_readiness_summary"]
        assert readiness["semantic_grounding_ready"] is True

    def test_semantic_grounding_not_ready_when_below(self):
        sem = _default_semantic_eval()
        sem["projection_exact_match"]["value"] = 0.40
        report = _base_report(semantic_evaluation=sem)
        report["test_performance"]["summary"]["projection_exact_match_rate"] = 0.40
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        readiness = result["production_readiness_summary"]
        assert readiness["semantic_grounding_ready"] is False


class TestApplicabilityCaseMinimum:
    def test_low_support_warns_but_does_not_block(self):
        sem = _default_semantic_eval()
        sem["filter_column_accuracy"]["value"] = 0.40
        sem["filter_column_accuracy"]["applicable_cases"] = 10  # Below 50 minimum
        report = _base_report(semantic_evaluation=sem)
        gate = ModelQualityGate()
        result = gate.evaluate(report, _thresholds())
        # Should warn but NOT block because support is too low
        failed_metrics = [c["metric"] for c in result["failed_checks"]]
        assert "filter_column_accuracy_rate" not in failed_metrics
        # Should have a warning about insufficient support
        low_support_warnings = [
            w for w in result["warnings"]
            if "applicable cases" in w and "filter_column_accuracy_rate" in w
        ]
        assert len(low_support_warnings) == 1
