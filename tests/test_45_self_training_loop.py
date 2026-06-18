"""Tests for self_training.self_training_loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_training.self_training_loop import SelfTrainingConfig, SelfTrainingLoop


@pytest.fixture
def config(tmp_path):
    """Config with tmp paths so tests don't touch real data."""
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"

    # Write minimal training/validation/test data
    examples = [
        {
            "example_id": f"ex_{i}",
            "question": f"question {i}",
            "dataset_name": "test",
            "db_id": "test_db",
            "split": "train",
            "query_ir": {"intent": "show_records", "base_table": "orders"},
            "source_sql": "SELECT * FROM orders",
            "schema": {"tables": {"orders": {"columns": ["id", "amount"]}}},
            "serialized_schema": "orders(id, amount)",
        }
        for i in range(5)
    ]

    for path in [train, val, test]:
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")

    return SelfTrainingConfig(
        train_path=train,
        validation_path=val,
        test_path=test,
        model_output_dir=tmp_path / "model",
        artifacts_dir=tmp_path / "artifacts",
        max_iterations=1,
        epochs_per_iteration=1,
        batch_size=2,
        max_prediction_examples=3,
    )


class TestSelfTrainingConfig:
    def test_defaults(self):
        cfg = SelfTrainingConfig()
        assert cfg.max_iterations == 3
        assert cfg.min_improvement == 0.005
        assert cfg.correction_weight == 2.0
        assert cfg.use_hard_negatives is True
        assert cfg.use_corrections is True

    def test_custom_values(self):
        cfg = SelfTrainingConfig(max_iterations=5, min_improvement=0.01)
        assert cfg.max_iterations == 5
        assert cfg.min_improvement == 0.01


class TestSelfTrainingLoop:
    def test_init(self, config):
        loop = SelfTrainingLoop(config)
        assert loop.config.max_iterations == 1

    def test_run_no_model(self, config):
        """When no model.pt exists, the loop should handle gracefully."""
        loop = SelfTrainingLoop(config)
        # The loop will try to train from scratch but will likely fail
        # since there's no real training infrastructure in the test.
        # It should still return a report dict without crashing.
        report = loop.run()
        assert isinstance(report, dict)

    def test_run_empty_data(self, tmp_path):
        """When training data is empty, the loop should return early."""
        empty_path = tmp_path / "empty.jsonl"
        empty_path.write_text("", encoding="utf-8")

        cfg = SelfTrainingConfig(
            train_path=empty_path,
            validation_path=empty_path,
            test_path=empty_path,
            model_output_dir=tmp_path / "model",
            artifacts_dir=tmp_path / "artifacts",
        )
        loop = SelfTrainingLoop(cfg)
        report = loop.run()
        assert isinstance(report, dict)

    def test_artifacts_dir_created(self, config):
        loop = SelfTrainingLoop(config)
        loop.run()
        assert config.artifacts_dir.exists()


class TestModelSelector:
    def test_select_best(self):
        from self_training.model_selector import ModelSelector
        selector = ModelSelector()
        iterations = [
            {"overall_slot_accuracy": 0.50},
            {"overall_slot_accuracy": 0.70},
            {"overall_slot_accuracy": 0.65},
        ]
        assert selector.select_best(iterations) == 1

    def test_select_best_empty(self):
        from self_training.model_selector import ModelSelector
        selector = ModelSelector()
        assert selector.select_best([]) == 0

    def test_compare_models(self):
        from self_training.model_selector import ModelSelector
        selector = ModelSelector()
        a = {"overall_slot_accuracy": 0.60, "exact_match_rate": 0.30}
        b = {"overall_slot_accuracy": 0.70, "exact_match_rate": 0.40}
        result = selector.compare_models(a, b)
        assert result["winner"] == "model_b"
        assert result["metric_diff"] > 0

    def test_promote_best(self, tmp_path):
        from self_training.model_selector import ModelSelector
        selector = ModelSelector()
        src = tmp_path / "src"
        src.mkdir()
        (src / "model.pt").write_text("fake")
        (src / "config.json").write_text("{}")
        dst = tmp_path / "dst"
        result = selector.promote_best(src, dst)
        assert result["success"]
        assert (dst / "model.pt").exists()
        assert (dst / "promotion_metadata.json").exists()
