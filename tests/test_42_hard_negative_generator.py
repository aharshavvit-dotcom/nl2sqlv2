"""Tests for self_training.hard_negative_generator."""

from __future__ import annotations

import pytest

from self_training.error_classifier import ErrorCategory, ErrorClassification
from self_training.hard_negative_generator import PredictionHardNegativeGenerator


@pytest.fixture
def generator():
    return PredictionHardNegativeGenerator()


def _classification(example_id: str, categories: list[ErrorCategory]) -> ErrorClassification:
    return ErrorClassification(example_id=example_id, categories=categories, severity="major")


def _example(example_id: str, gold_ir: dict, pred_ir: dict) -> dict:
    return {
        "example_id": example_id,
        "question": "test question",
        "dataset_name": "test",
        "db_id": "test_db",
        "query_ir": gold_ir,
        "predicted_query_ir": pred_ir,
    }


class TestGenerateFromErrors:
    def test_wrong_intent_negative(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT])
        ex = _example("1", {"intent": "metric_summary"}, {"intent": "show_records"})
        negatives = generator.generate_from_errors([ec], [ex])
        assert len(negatives) >= 1
        neg = negatives[0]
        assert neg["negative_type"] == "wrong_intent"
        assert neg["source"] == "prediction_error"
        assert neg["negative_query_ir"]["intent"] == "show_records"
        assert neg["gold_query_ir"]["intent"] == "metric_summary"

    def test_wrong_base_table_negative(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_BASE_TABLE])
        ex = _example("1", {"base_table": "orders"}, {"base_table": "products"})
        negatives = generator.generate_from_errors([ec], [ex])
        assert len(negatives) >= 1
        assert negatives[0]["negative_query_ir"]["base_table"] == "products"

    def test_unnecessary_join_negative(self, generator):
        ec = _classification("1", [ErrorCategory.UNNECESSARY_JOIN])
        ex = _example(
            "1",
            {"base_table": "orders", "joins": []},
            {"base_table": "orders", "joins": [{"condition": "fake join"}]},
        )
        negatives = generator.generate_from_errors([ec], [ex])
        assert len(negatives) >= 1
        assert negatives[0]["negative_type"] == "unnecessary_join"

    def test_multiple_error_categories(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT, ErrorCategory.WRONG_BASE_TABLE])
        ex = _example("1", {"intent": "a", "base_table": "t1"}, {"intent": "b", "base_table": "t2"})
        negatives = generator.generate_from_errors([ec], [ex])
        assert len(negatives) == 2
        types = {n["negative_type"] for n in negatives}
        assert "wrong_intent" in types
        assert "wrong_base_table" in types

    def test_empty_input(self, generator):
        negatives = generator.generate_from_errors([], [])
        assert negatives == []

    def test_no_predicted_ir_skipped(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT])
        ex = {"example_id": "1", "query_ir": {"intent": "a"}, "predicted_query_ir": {}}
        negatives = generator.generate_from_errors([ec], [ex])
        # Empty predicted_query_ir still produces a result since intent differs
        # (gold has "a", pred has None effectively)
        assert isinstance(negatives, list)

    def test_output_format(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_METRIC])
        ex = _example(
            "1",
            {"metrics": [{"aggregation": "SUM", "expression": "amount"}]},
            {"metrics": [{"aggregation": "COUNT", "expression": "id"}]},
        )
        negatives = generator.generate_from_errors([ec], [ex])
        assert len(negatives) >= 1
        neg = negatives[0]
        assert "example_id" in neg
        assert "negative_id" in neg
        assert "question" in neg
        assert "gold_query_ir" in neg
        assert "negative_query_ir" in neg
        assert "negative_type" in neg


class TestContrastivePairs:
    def test_contrastive_pair_generation(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT])
        ex = _example("1", {"intent": "a"}, {"intent": "b"})
        pairs = generator.generate_contrastive_pairs([ec], [ex])
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["source"] == "contrastive_pair"
        assert pair["gold_query_ir"]["intent"] == "a"
        assert pair["predicted_query_ir"]["intent"] == "b"
        assert "wrong_intent" in pair["error_categories"]

    def test_empty_classifications(self, generator):
        pairs = generator.generate_contrastive_pairs([], [])
        assert pairs == []
