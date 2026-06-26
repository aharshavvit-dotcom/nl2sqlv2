"""Smoke test for the optimized neural training loop.

Uses mock data to verify the training pipeline runs end-to-end
without errors.
"""

import json
from pathlib import Path

import pytest
import torch

from dataset_training.curriculum_builder import CurriculumBuilder
from neural_optimization.training_config import NeuralTrainingConfig
from neural_optimization.checkpoint_manager import CheckpointManager
from neural_optimization.training_diagnostics import TrainingDiagnostics
from neural_optimization.loss_weighter import MultiTaskLossWeighter
from neural_optimization.early_stopping import EarlyStopping


class TestOptimizedTrainingSmoke:
    def test_diagnostics_record_and_save(self, tmp_path):
        diag = TrainingDiagnostics(tmp_path)
        diag.set_config({"optimizer": {"name": "adamw"}, "model": {"activation": "gelu"}})
        diag.start_training()
        diag.record_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "overall_slot_accuracy": 0.6},
            val_metrics={"loss": 0.4, "overall_slot_accuracy": 0.65},
            lr=0.0007,
            epoch_time=10.0,
        )
        diag.save(tmp_path)
        assert (tmp_path / "training_diagnostics.json").exists()
        assert (tmp_path / "training_diagnostics.md").exists()
        data = json.loads((tmp_path / "training_diagnostics.json").read_text(encoding="utf-8"))
        assert data["total_epochs"] == 1
        assert data["best_epoch"] == 1

    def test_checkpoint_and_early_stopping_integration(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        mgr = CheckpointManager(tmp_path, metric_name="accuracy", mode="max")
        es = EarlyStopping(patience=2, metric_name="accuracy", mode="max")

        # Epoch 1: best
        mgr.maybe_save_best(model, opt, 1, {"accuracy": 0.7})
        es.step({"accuracy": 0.7})
        # Epoch 2: better
        mgr.maybe_save_best(model, opt, 2, {"accuracy": 0.8})
        es.step({"accuracy": 0.8})
        # Epoch 3: worse
        mgr.maybe_save_best(model, opt, 3, {"accuracy": 0.75})
        es.step({"accuracy": 0.75})
        # Epoch 4: worse
        assert es.step({"accuracy": 0.74}) is True

        loaded = mgr.load_best()
        assert loaded["epoch"] == 2

    def test_loss_weighter_with_diagnostics(self, tmp_path):
        weighter = MultiTaskLossWeighter({"intent": 1.0, "base_table": 1.2})
        losses = {
            "intent": torch.tensor(0.5),
            "base_table": torch.tensor(0.3),
        }
        combined = weighter.combine(losses)
        diag = TrainingDiagnostics(tmp_path)
        diag.start_training()
        diag.record_epoch(
            epoch=1,
            train_metrics={"loss": combined["total_loss"].item()},
            val_metrics={"loss": 0.3, "overall_slot_accuracy": 0.7},
            loss_by_head=combined["raw_losses"],
        )
        result = diag.to_dict()
        assert result["epochs"][0]["loss_by_head"]["intent"] == 0.5

    def test_config_smoke(self):
        cfg = NeuralTrainingConfig()
        assert cfg.model["feed_forward_heads"] is True
        assert cfg.optimizer["name"] == "adamw"
        assert cfg.training["gradient_clipping"] == 1.0

    def test_ordered_dataset_curriculum_reports_not_phased(self):
        rows = [{"query_ir": {"intent": "show_records"}}]
        ordered, distribution = CurriculumBuilder().order_examples(rows, mode="ordered_dataset")

        assert ordered == rows
        assert distribution["_curriculum_mode"] == "ordered_dataset"
        assert distribution["_phased_epochs"] is False

    def test_phased_epochs_curriculum_fails_without_explicit_fallback(self):
        with pytest.raises(NotImplementedError, match="phased_epochs curriculum requested but not implemented"):
            CurriculumBuilder().order_examples([], mode="phased_epochs")

    def test_phased_epochs_curriculum_fallback_must_be_explicit(self):
        ordered, distribution = CurriculumBuilder().order_examples(
            [{"query_ir": {"intent": "show_records"}}],
            mode="phased_epochs",
            allow_ordered_dataset_fallback=True,
        )

        assert len(ordered) == 1
        assert distribution["_curriculum_mode"] == "ordered_dataset"
