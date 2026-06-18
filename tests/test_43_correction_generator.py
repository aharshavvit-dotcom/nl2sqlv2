"""Tests for self_training.correction_generator."""

from __future__ import annotations

import pytest

from self_training.correction_generator import CorrectionExampleGenerator
from self_training.error_classifier import ErrorCategory, ErrorClassification


@pytest.fixture
def generator():
    return CorrectionExampleGenerator(correction_weight=2.0)


def _classification(example_id: str, categories: list[ErrorCategory]) -> ErrorClassification:
    return ErrorClassification(
        example_id=example_id,
        categories=categories,
        severity="major",
        suggested_fix_type="intent_correction",
    )


def _example(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "question": "Show total revenue",
        "dataset_name": "test",
        "db_id": "test_db",
        "split": "train",
        "source_sql": "SELECT SUM(amount) FROM orders",
        "query_ir": {"intent": "metric_summary", "base_table": "orders"},
        "predicted_query_ir": {"intent": "show_records", "base_table": "orders"},
    }


class TestGenerate:
    def test_correction_generated(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT])
        corrections = generator.generate([ec], [_example("1")])
        assert len(corrections) == 1
        corr = corrections[0]
        assert corr["original_example_id"] == "1"
        assert corr["correction_type"] == "intent_correction"
        assert corr["query_ir"]["intent"] == "metric_summary"  # gold
        assert corr["wrong_prediction"]["intent"] == "show_records"  # wrong

    def test_correction_weight(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT])
        corrections = generator.generate([ec], [_example("1")])
        assert corrections[0]["metadata"]["correction_weight"] == 2.0

    def test_empty_input(self, generator):
        corrections = generator.generate([], [])
        assert corrections == []

    def test_no_categories_skipped(self, generator):
        ec = ErrorClassification(example_id="1", categories=[], severity="minor")
        corrections = generator.generate([ec], [_example("1")])
        assert corrections == []

    def test_no_gold_ir_skipped(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_INTENT])
        example = {"example_id": "1", "question": "test"}
        corrections = generator.generate([ec], [example])
        assert corrections == []

    def test_correction_fields(self, generator):
        ec = _classification("1", [ErrorCategory.WRONG_BASE_TABLE])
        corrections = generator.generate([ec], [_example("1")])
        corr = corrections[0]
        assert "example_id" in corr
        assert "question" in corr
        assert "dataset_name" in corr
        assert "error_categories" in corr
        assert "severity" in corr
        assert "metadata" in corr
        assert corr["metadata"]["source"] == "correction"


class TestAugmentedTrainingSet:
    def test_merge_all_sources(self, generator):
        original = [{"example_id": "o1", "query_ir": {"intent": "a"}}]
        corrections = [{"example_id": "c1", "query_ir": {"intent": "b"}}]
        hard_negatives = [
            {
                "example_id": "n1",
                "negative_id": "n1_neg",
                "question": "q",
                "gold_query_ir": {"intent": "c"},
                "negative_query_ir": {"intent": "d"},
                "negative_type": "wrong_intent",
            }
        ]
        augmented = generator.generate_augmented_training_set(original, corrections, hard_negatives)
        assert len(augmented) == 3

        # Check sources
        sources = [row.get("metadata", {}).get("source") for row in augmented]
        assert "original" in sources
        assert "correction" in sources
        assert "hard_negative" in sources

    def test_weights_assigned(self, generator):
        original = [{"example_id": "o1"}]
        corrections = [{"example_id": "c1", "metadata": {"correction_weight": 2.0}}]
        augmented = generator.generate_augmented_training_set(original, corrections, [])
        weights = [row.get("metadata", {}).get("correction_weight") for row in augmented]
        assert 1.0 in weights  # original
        assert 2.0 in weights  # correction

    def test_empty_inputs(self, generator):
        augmented = generator.generate_augmented_training_set([], [], [])
        assert augmented == []

    def test_original_not_mutated(self, generator):
        original = [{"example_id": "o1", "query_ir": {"intent": "a"}}]
        original_copy = [dict(original[0])]
        generator.generate_augmented_training_set(original, [], [])
        assert original[0]["example_id"] == original_copy[0]["example_id"]
