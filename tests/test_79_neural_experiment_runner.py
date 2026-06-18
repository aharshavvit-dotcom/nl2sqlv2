"""Tests for neural_optimization.experiment_runner and experiment_reporter."""

from __future__ import annotations

import json
import yaml
from pathlib import Path

from neural_optimization.training_config import NeuralTrainingConfig
from neural_optimization.experiment_runner import ExperimentRunner, load_experiment_grid
from neural_optimization.experiment_reporter import ExperimentReporter


class TestExperimentRunner:
    def test_run_grid(self, tmp_path):
        base = NeuralTrainingConfig()
        grid = [
            {"name": "exp_a", "optimizer": {"name": "adam"}},
            {"name": "exp_b", "optimizer": {"name": "adamw"}},
        ]
        runner = ExperimentRunner(base, grid, tmp_path)

        def mock_train(config, out_dir):
            return {"overall_slot_accuracy": 0.7 if config.optimizer["name"] == "adamw" else 0.5}

        results = runner.run(mock_train)
        assert len(results) == 2
        assert results[0]["name"] == "exp_a"
        assert results[1]["name"] == "exp_b"

    def test_experiment_failure_handled(self, tmp_path):
        base = NeuralTrainingConfig()
        grid = [{"name": "fail_exp"}]
        runner = ExperimentRunner(base, grid, tmp_path)

        def fail_train(config, out_dir):
            raise RuntimeError("deliberate failure")

        results = runner.run(fail_train)
        assert "error" in results[0]["metrics"]


class TestExperimentReporter:
    def test_best_experiment(self):
        results = [
            {"name": "a", "metrics": {"overall_slot_accuracy": 0.6}},
            {"name": "b", "metrics": {"overall_slot_accuracy": 0.8}},
        ]
        reporter = ExperimentReporter(results)
        best = reporter.best_experiment()
        assert best["name"] == "b"

    def test_summary_structure(self):
        results = [
            {"name": "a", "metrics": {"overall_slot_accuracy": 0.6}},
        ]
        reporter = ExperimentReporter(results)
        summary = reporter.summary()
        assert summary["total_experiments"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0

    def test_save_reports(self, tmp_path):
        results = [
            {"name": "a", "optimizer": "adam", "activation": "gelu",
             "learning_rate": 0.001, "training_time_seconds": 10.0,
             "metrics": {"overall_slot_accuracy": 0.7}},
        ]
        reporter = ExperimentReporter(results)
        reporter.save(tmp_path)
        assert (tmp_path / "experiment_summary.json").exists()
        assert (tmp_path / "experiment_summary.md").exists()


class TestLoadExperimentGrid:
    def test_load_from_yaml(self, tmp_path):
        grid_file = tmp_path / "grid.yaml"
        grid_file.write_text(yaml.dump({
            "experiments": [
                {"name": "exp1", "optimizer": {"name": "adam"}},
                {"name": "exp2", "optimizer": {"name": "sgd"}},
            ]
        }), encoding="utf-8")
        grid = load_experiment_grid(grid_file)
        assert len(grid) == 2
        assert grid[0]["name"] == "exp1"
