"""Tests for Gate 5: Evaluation + RAG + Confidence."""

from __future__ import annotations

import pytest

from evaluation.evaluation_framework import (
    ComplexitySlice,
    ConfidenceScores,
    MetricResult,
    SeedRunResult,
    StatisticalReporter,
    FrozenSplitAccessError,
    FrozenSplitGuard,
    component_accuracy,
    exact_match_accuracy,
)


class TestExactMatch:
    def test_perfect_match(self):
        preds = [{"source_sql": "SELECT id FROM orders LIMIT 10"}]
        gold = [{"source_sql": "SELECT id FROM orders LIMIT 10"}]
        result = exact_match_accuracy(preds, gold)
        assert result.value == 1.0

    def test_case_insensitive(self):
        preds = [{"source_sql": "select id from orders limit 10"}]
        gold = [{"source_sql": "SELECT id FROM orders LIMIT 10"}]
        result = exact_match_accuracy(preds, gold)
        assert result.value == 1.0

    def test_partial_match(self):
        preds = [
            {"source_sql": "SELECT id FROM orders LIMIT 10"},
            {"source_sql": "SELECT name FROM wrong LIMIT 10"},
        ]
        gold = [
            {"source_sql": "SELECT id FROM orders LIMIT 10"},
            {"source_sql": "SELECT name FROM customers LIMIT 10"},
        ]
        result = exact_match_accuracy(preds, gold)
        assert result.value == 0.5


class TestComplexitySlice:
    def test_complexity_classification(self):
        simple = {"query_ir": {"required_tables": ["t1"]}}
        moderate = {"query_ir": {"joins": [{"join": 1}], "where": {"col": "x"}}}
        complex_ = {"query_ir": {"joins": [1, 2], "ctes": [1], "having": True}}

        slicer = ComplexitySlice()
        # Just verify the slicer can process without errors
        preds = [simple, moderate, complex_]
        gold = [simple, moderate, complex_]
        result = slicer.evaluate(preds, gold)
        assert "simple" in result
        assert "moderate" in result
        assert "complex" in result


class TestStatisticalReporter:
    def test_multi_seed_report(self):
        reporter = StatisticalReporter()
        reporter.add_run(SeedRunResult(seed=1, metrics={"em": 0.85}))
        reporter.add_run(SeedRunResult(seed=2, metrics={"em": 0.87}))
        reporter.add_run(SeedRunResult(seed=3, metrics={"em": 0.83}))
        report = reporter.report()
        assert report["num_seeds"] == 3
        assert "em" in report["metrics"]
        em = report["metrics"]["em"]
        assert 0.83 <= em["mean"] <= 0.87
        assert em["ci"][0] < em["mean"] < em["ci"][1]

    def test_single_seed(self):
        reporter = StatisticalReporter()
        reporter.add_run(SeedRunResult(seed=42, metrics={"em": 0.90}))
        report = reporter.report()
        assert report["metrics"]["em"]["mean"] == 0.90


class TestFrozenSplitGuard:
    def test_frozen_split_blocked(self):
        guard = FrozenSplitGuard()
        with pytest.raises(FrozenSplitAccessError):
            guard.check_access("frozen_semantic_test", "training")

    def test_train_split_allowed(self):
        guard = FrozenSplitGuard()
        guard.check_access("train", "training")  # Should not raise

    def test_allowed_evaluation_access(self):
        guard = FrozenSplitGuard()
        guard.allow_evaluation_access("frozen_semantic_test")
        guard.check_access("frozen_semantic_test", "final_eval")  # Should work

    def test_is_frozen(self):
        guard = FrozenSplitGuard()
        assert guard.is_frozen("frozen_semantic_test") is True
        assert guard.is_frozen("train") is False


class TestConfidenceScores:
    def test_actionable_prediction(self):
        conf = ConfidenceScores(model_confidence=0.9, capability_coverage=0.95, safety_score=0.98)
        assert conf.is_actionable is True
        assert conf.overall > 0.8

    def test_low_confidence_not_actionable(self):
        conf = ConfidenceScores(model_confidence=0.3, capability_coverage=0.95, safety_score=0.98)
        assert conf.is_actionable is False

    def test_unsafe_not_actionable(self):
        conf = ConfidenceScores(model_confidence=0.9, capability_coverage=0.95, safety_score=0.5)
        assert conf.is_actionable is False

    def test_serialization(self):
        conf = ConfidenceScores(model_confidence=0.9, capability_coverage=0.8, safety_score=1.0)
        d = conf.to_dict()
        assert "model_confidence" in d
        assert "is_actionable" in d
