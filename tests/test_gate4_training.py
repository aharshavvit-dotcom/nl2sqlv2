"""Tests for Gate 4: Training Readiness."""

from __future__ import annotations

import pytest

from training.multi_task_loss import (
    CheckpointMetrics,
    CheckpointSelector,
    HardFloorConfig,
    MultiTaskLossComputer,
    MultiTaskLossConfig,
    TaskLossConfig,
    TrainingConfig,
)


class TestMultiTaskLoss:
    def test_uncertainty_weighting(self):
        config = MultiTaskLossConfig(
            tasks=[
                TaskLossConfig(name="query_ir", weight_floor=0.1),
                TaskLossConfig(name="capability", weight_floor=0.05),
            ],
            weighting_strategy="uncertainty",
        )
        computer = MultiTaskLossComputer(config)
        result = computer.compute({"query_ir": 1.5, "capability": 0.8})
        assert result["total_loss"] > 0
        assert "query_ir" in result["task_contributions"]
        assert "capability" in result["task_contributions"]

    def test_fixed_weighting(self):
        config = MultiTaskLossConfig(
            tasks=[
                TaskLossConfig(name="query_ir", weight_floor=0.5),
                TaskLossConfig(name="capability", weight_floor=0.3),
            ],
            weighting_strategy="fixed",
        )
        computer = MultiTaskLossComputer(config)
        result = computer.compute({"query_ir": 2.0, "capability": 1.0})
        assert abs(result["effective_weights"]["query_ir"] - 0.5) < 1e-6
        assert abs(result["effective_weights"]["capability"] - 0.3) < 1e-6

    def test_weight_floor_enforced(self):
        config = MultiTaskLossConfig(
            tasks=[
                TaskLossConfig(name="task_a", weight_floor=0.1, initial_log_variance=5.0),
            ],
            weighting_strategy="uncertainty",
        )
        computer = MultiTaskLossComputer(config)
        result = computer.compute({"task_a": 1.0})
        assert result["effective_weights"]["task_a"] >= 0.1

    def test_disabled_task_ignored(self):
        config = MultiTaskLossConfig(
            tasks=[
                TaskLossConfig(name="enabled", enabled=True),
                TaskLossConfig(name="disabled", enabled=False),
            ],
        )
        computer = MultiTaskLossComputer(config)
        result = computer.compute({"enabled": 1.0, "disabled": 1.0})
        assert "enabled" in result["task_contributions"]
        assert "disabled" not in result["task_contributions"]

    def test_invalid_config_raises(self):
        config = MultiTaskLossConfig(tasks=[])
        with pytest.raises(ValueError, match="No tasks"):
            MultiTaskLossComputer(config)


class TestCheckpointSelector:
    def test_selects_best_passing_checkpoint(self):
        floors = HardFloorConfig(metric_floors={"accuracy": 0.8, "safety": 0.9})
        selector = CheckpointSelector(floors)
        selector.add_checkpoint(CheckpointMetrics(
            epoch=1, step=100, primary_metric=0.85,
            task_metrics={"accuracy": 0.82, "safety": 0.95},
        ))
        selector.add_checkpoint(CheckpointMetrics(
            epoch=2, step=200, primary_metric=0.90,
            task_metrics={"accuracy": 0.75, "safety": 0.95},  # Fails accuracy floor
        ))
        selector.add_checkpoint(CheckpointMetrics(
            epoch=3, step=300, primary_metric=0.88,
            task_metrics={"accuracy": 0.85, "safety": 0.92},
        ))
        best = selector.select_best()
        assert best is not None
        assert best.epoch == 3  # Best primary among eligible

    def test_no_eligible_returns_none(self):
        floors = HardFloorConfig(metric_floors={"accuracy": 0.99})
        selector = CheckpointSelector(floors)
        selector.add_checkpoint(CheckpointMetrics(
            epoch=1, step=100, primary_metric=0.9,
            task_metrics={"accuracy": 0.5},
        ))
        assert selector.select_best() is None

    def test_report_includes_violations(self):
        floors = HardFloorConfig(metric_floors={"accuracy": 0.8})
        selector = CheckpointSelector(floors)
        selector.add_checkpoint(CheckpointMetrics(
            epoch=1, step=100, primary_metric=0.9,
            task_metrics={"accuracy": 0.7},
        ))
        report = selector.selection_report()
        assert report["eligible_count"] == 0
        assert len(report["checkpoints"][0]["violations"]) > 0


class TestTrainingConfig:
    def test_valid_config(self):
        config = TrainingConfig(
            multi_task=MultiTaskLossConfig(
                tasks=[TaskLossConfig(name="query_ir")],
            ),
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_same_splits_rejected(self):
        config = TrainingConfig(
            validation_split="frozen_semantic_test",
            test_split="frozen_semantic_test",
            multi_task=MultiTaskLossConfig(
                tasks=[TaskLossConfig(name="query_ir")],
            ),
        )
        errors = config.validate()
        assert any("validation_split must differ" in e for e in errors)
