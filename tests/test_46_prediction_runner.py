"""Tests for self_training.prediction_runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from self_training.prediction_runner import PredictionRunner


@pytest.fixture
def mock_predictor():
    """Create a mock NeuralIRPredictor."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "query_ir": {"intent": "show_records", "base_table": "orders"},
        "sql": "SELECT * FROM orders",
        "confidence": 0.85,
        "raw_confidence": 0.82,
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "warnings": [],
    }
    return predictor


class TestPredictBatch:
    @patch("self_training.prediction_runner.PredictionRunner._load_predictor")
    def test_batch_prediction(self, mock_load, mock_predictor):
        mock_load.return_value = mock_predictor
        runner = PredictionRunner("fake/model/dir")

        examples = [
            {
                "example_id": "1",
                "question": "show orders",
                "dataset_name": "test",
                "db_id": "test_db",
                "query_ir": {"intent": "show_records"},
                "source_sql": "SELECT * FROM orders",
                "schema": {"tables": {"orders": {}}},
            },
            {
                "example_id": "2",
                "question": "count products",
                "dataset_name": "test",
                "db_id": "test_db",
                "query_ir": {"intent": "count_records"},
                "source_sql": "SELECT COUNT(*) FROM products",
                "schema": {"tables": {"products": {}}},
            },
        ]

        results = runner.predict_batch(examples)
        assert len(results) == 2
        assert results[0]["example_id"] == "1"
        assert results[0]["prediction_failed"] is False
        assert results[0]["predicted_query_ir"] is not None
        assert results[0]["gold_query_ir"]["intent"] == "show_records"

    @patch("self_training.prediction_runner.PredictionRunner._load_predictor")
    def test_max_examples(self, mock_load, mock_predictor):
        mock_load.return_value = mock_predictor
        runner = PredictionRunner("fake/model/dir")

        examples = [{"example_id": str(i), "question": f"q{i}", "schema": {}} for i in range(10)]
        results = runner.predict_batch(examples, max_examples=3)
        assert len(results) == 3

    @patch("self_training.prediction_runner.PredictionRunner._load_predictor")
    def test_prediction_failure_handled(self, mock_load):
        failing_predictor = MagicMock()
        failing_predictor.predict.side_effect = RuntimeError("model crashed")
        mock_load.return_value = failing_predictor

        runner = PredictionRunner("fake/model/dir")
        examples = [{"example_id": "1", "question": "test", "schema": {}}]
        results = runner.predict_batch(examples)

        assert len(results) == 1
        assert results[0]["prediction_failed"] is True
        assert "model crashed" in results[0]["error_message"]
        assert results[0]["confidence"] == 0.0

    @patch("self_training.prediction_runner.PredictionRunner._load_predictor")
    def test_timing_recorded(self, mock_load, mock_predictor):
        mock_load.return_value = mock_predictor
        runner = PredictionRunner("fake/model/dir")
        examples = [{"example_id": "1", "question": "test", "schema": {}}]
        results = runner.predict_batch(examples)
        assert results[0]["prediction_time_ms"] >= 0

    @patch("self_training.prediction_runner.PredictionRunner._load_predictor")
    def test_output_fields(self, mock_load, mock_predictor):
        mock_load.return_value = mock_predictor
        runner = PredictionRunner("fake/model/dir")
        examples = [
            {
                "example_id": "1",
                "question": "test",
                "dataset_name": "wikisql",
                "db_id": "db1",
                "split": "train",
                "query_ir": {"intent": "a"},
                "source_sql": "SELECT 1",
                "schema": {},
            }
        ]
        results = runner.predict_batch(examples)
        r = results[0]
        required_fields = [
            "example_id", "question", "dataset_name", "db_id", "split",
            "predicted_query_ir", "predicted_sql", "confidence",
            "gold_query_ir", "gold_sql", "prediction_failed",
            "error_message", "prediction_time_ms",
        ]
        for field in required_fields:
            assert field in r, f"Missing field: {field}"
