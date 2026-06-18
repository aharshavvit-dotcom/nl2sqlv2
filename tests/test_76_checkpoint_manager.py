"""Tests for neural_optimization.checkpoint_manager."""

from __future__ import annotations

import json
import torch
from torch import nn

from neural_optimization.checkpoint_manager import CheckpointManager


def _dummy_model():
    return nn.Linear(4, 2)


def _dummy_optimizer(model):
    return torch.optim.Adam(model.parameters())


class TestCheckpointManager:
    def test_save_best(self, tmp_path):
        model = _dummy_model()
        opt = _dummy_optimizer(model)
        mgr = CheckpointManager(tmp_path, metric_name="accuracy", mode="max")
        saved = mgr.maybe_save_best(model, opt, epoch=1, metrics={"accuracy": 0.8})
        assert saved is True
        assert (tmp_path / "best_model.pt").exists()

    def test_no_save_when_worse(self, tmp_path):
        model = _dummy_model()
        opt = _dummy_optimizer(model)
        mgr = CheckpointManager(tmp_path, metric_name="accuracy", mode="max")
        mgr.maybe_save_best(model, opt, epoch=1, metrics={"accuracy": 0.8})
        saved = mgr.maybe_save_best(model, opt, epoch=2, metrics={"accuracy": 0.7})
        assert saved is False

    def test_save_last(self, tmp_path):
        model = _dummy_model()
        opt = _dummy_optimizer(model)
        mgr = CheckpointManager(tmp_path)
        mgr.save_last(model, opt, epoch=1, metrics={"loss": 0.5})
        assert (tmp_path / "last_model.pt").exists()

    def test_metadata_written(self, tmp_path):
        model = _dummy_model()
        opt = _dummy_optimizer(model)
        mgr = CheckpointManager(tmp_path, metric_name="accuracy", mode="max")
        mgr.maybe_save_best(model, opt, epoch=1, metrics={"accuracy": 0.9})
        meta_path = tmp_path / "checkpoint_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["best_epoch"] == 1
        assert meta["best_metric_value"] == 0.9

    def test_load_best(self, tmp_path):
        model = _dummy_model()
        opt = _dummy_optimizer(model)
        mgr = CheckpointManager(tmp_path, metric_name="accuracy", mode="max")
        mgr.maybe_save_best(model, opt, epoch=1, metrics={"accuracy": 0.9})
        loaded = mgr.load_best()
        assert loaded is not None
        assert loaded["epoch"] == 1
        assert "model_state_dict" in loaded

    def test_load_best_missing(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        assert mgr.load_best() is None

    def test_min_mode(self, tmp_path):
        model = _dummy_model()
        opt = _dummy_optimizer(model)
        mgr = CheckpointManager(tmp_path, metric_name="loss", mode="min")
        mgr.maybe_save_best(model, opt, epoch=1, metrics={"loss": 0.5})
        saved = mgr.maybe_save_best(model, opt, epoch=2, metrics={"loss": 0.3})
        assert saved is True
        saved = mgr.maybe_save_best(model, opt, epoch=3, metrics={"loss": 0.4})
        assert saved is False
