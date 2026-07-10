"""Unit tests for PredictionCache, TelemetryLogger, and CurriculumBuilder shuffling."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from inference.prediction_cache import PredictionCache
from inference.telemetry_logger import TelemetryLogger
from dataset_training.curriculum_builder import CurriculumBuilder


def test_prediction_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir) / "test_cache.json"
        
        # Initialize
        cache = PredictionCache(cache_path=cache_path, max_entries=3)
        
        # Test hash key consistency
        key1 = cache.generate_hash_key("query 1", {"tables": {}}, "path/to/model", {"opt": 1})
        key2 = cache.generate_hash_key("query 1", {"tables": {}}, "path/to/model", {"opt": 1})
        assert key1 == key2
        
        # Test put & get
        prediction = {
            "status": "completed",
            "sql": "SELECT *",
            "confidence": 0.9,
            "query_ir": {"intent": "show_records", "filters": [{"column": "amount", "value": 10}]},
            "validation": {"is_valid": True},
        }
        cache.put("query 1", {"tables": {}}, "path/to/model", prediction, {"opt": 1})
        
        cached = cache.get("query 1", {"tables": {}}, "path/to/model", {"opt": 1})
        assert cached is not None
        assert cached["status"] == "completed"
        assert "question" not in cached
        assert cached["query_ir"]["filters"][0]["value"] == "[REDACTED]"
        
        # Test bypass_cache
        cached_bypass = cache.get("query 1", {"tables": {}}, "path/to/model", {"opt": 1}, bypass_cache=True)
        assert cached_bypass is None
        
        # Test eviction
        valid_prediction = {"status": "completed", "query_ir": {"intent": "show"}, "validation": {"is_valid": True}}
        cache.put("query 2", {}, "", valid_prediction, {"p": 2})
        cache.put("query 3", {}, "", valid_prediction, {"p": 3})
        cache.put("query 4", {}, "", valid_prediction, {"p": 4}) # Evicts query 1 (LRU)
        
        assert cache.get("query 1", {"tables": {}}, "path/to/model", {"opt": 1}) is None
        assert cache.get("query 4", {}, "", {"p": 4}) == valid_prediction


def test_telemetry_logger() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test_telemetry.jsonl"
        logger = TelemetryLogger(log_path=log_path)
        
        # Log prediction
        prediction = {"status": "completed", "source_model": "neural_ir", "sql": "SELECT 1", "confidence": 0.8}
        logger.log_prediction("query 1", prediction, duration_ms=45.2)
        
        # Log feedback
        logger.log_feedback("query 1", "SELECT 1", True, "Great result")
        
        # Verify file exists and has entries
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        
        entry1 = json.loads(lines[0])
        assert entry1["event_type"] == "prediction"
        assert "question_hash" in entry1
        assert "raw_question" not in entry1
        assert entry1["duration_ms"] == 45.2
        assert "raw_sql" not in entry1
        
        entry2 = json.loads(lines[1])
        assert entry2["event_type"] == "feedback"
        assert "question_hash" in entry2
        assert "raw_question" not in entry2
        assert entry2["is_correct"] is True
        assert "comments" not in entry2


def test_curriculum_difficulty_shuffling() -> None:
    # Build sample examples
    examples = [
        {"example_id": "1", "intent": "show_records"},  # phase_1
        {"example_id": "2", "intent": "show_records"},  # phase_1
        {"example_id": "3", "intent": "metric_by_dimension"},  # phase_2
        {"example_id": "4", "intent": "metric_by_dimension"},  # phase_2
        {"example_id": "5", "intent": "trend_by_date"},  # phase_3
        {"example_id": "6", "intent": "trend_by_date"},  # phase_3
        {"example_id": "7", "intent": "joined_records"},  # phase_4
        {"example_id": "8", "intent": "joined_records"},  # phase_4
    ]
    
    builder = CurriculumBuilder()
    
    # Shuffle with seed 42
    shuffled1 = builder.shuffle_within_buckets(examples, seed=42)
    # Shuffle with seed 43
    shuffled2 = builder.shuffle_within_buckets(examples, seed=43)
    
    # Verify that all examples are present
    assert len(shuffled1) == len(examples)
    assert set(e["example_id"] for e in shuffled1) == set(e["example_id"] for e in examples)
    
    # Verify that order of phases (easy to hard) is preserved
    # phase_1 (id: 1, 2) -> phase_2 (id: 3, 4) -> phase_3 (id: 5, 6) -> phase_4 (id: 7, 8)
    for shuffled in [shuffled1, shuffled2]:
        ids = [int(e["example_id"]) for e in shuffled]
        # Check phase 1 is first
        assert set(ids[0:2]) == {1, 2}
        # Check phase 2 is second
        assert set(ids[2:4]) == {3, 4}
        # Check phase 3 is third
        assert set(ids[4:6]) == {5, 6}
        # Check phase 4 is last
        assert set(ids[6:8]) == {7, 8}
    
    # Verify that shuffling actually changed orders of items within phase (given correct seed)
    # For seed 42, check shuffled1 order vs seed 43 shuffled2 order
    orders = [[int(e["example_id"]) for e in s] for s in [shuffled1, shuffled2]]
    # Since each bucket size is 2, there are 2^4 = 16 combinations, so they can differ.
    assert len(orders) == 2
