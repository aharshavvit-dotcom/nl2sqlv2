"""Tests for neural_optimization.loss_weighter."""

from __future__ import annotations

import torch
from neural_optimization.loss_weighter import MultiTaskLossWeighter


class TestMultiTaskLossWeighter:
    def test_weighted_combine(self):
        w = MultiTaskLossWeighter({"intent": 1.0, "base_table": 2.0})
        losses = {
            "intent": torch.tensor(0.5),
            "base_table": torch.tensor(0.3),
        }
        result = w.combine(losses)
        assert "total_loss" in result
        assert "weighted_losses" in result
        assert "raw_losses" in result
        # total = 1.0*0.5 + 2.0*0.3 = 1.1
        assert abs(result["total_loss"].item() - 1.1) < 1e-5

    def test_missing_head_ignored(self):
        w = MultiTaskLossWeighter({"intent": 1.0, "missing_head": 3.0})
        losses = {"intent": torch.tensor(0.5)}
        result = w.combine(losses)
        assert abs(result["total_loss"].item() - 0.5) < 1e-5

    def test_hard_negative_weight(self):
        w = MultiTaskLossWeighter({"intent": 1.0, "hard_negative": 0.3})
        losses = {
            "intent": torch.tensor(1.0),
            "hard_negative": torch.tensor(2.0),
        }
        result = w.combine(losses)
        # total = 1.0*1.0 + 0.3*2.0 = 1.6
        assert abs(result["total_loss"].item() - 1.6) < 1e-5

    def test_default_weight_is_one(self):
        w = MultiTaskLossWeighter({})
        losses = {"some_head": torch.tensor(0.7)}
        result = w.combine(losses)
        assert abs(result["total_loss"].item() - 0.7) < 1e-5

    def test_empty_losses(self):
        w = MultiTaskLossWeighter({"intent": 1.0})
        result = w.combine({})
        assert result["total_loss"].item() == 0.0

    def test_none_loss_skipped(self):
        w = MultiTaskLossWeighter({"intent": 1.0})
        result = w.combine({"intent": None, "base_table": torch.tensor(0.5)})
        assert abs(result["total_loss"].item() - 0.5) < 1e-5

    def test_raw_losses_are_floats(self):
        w = MultiTaskLossWeighter({"a": 2.0})
        result = w.combine({"a": torch.tensor(0.3)})
        assert isinstance(result["raw_losses"]["a"], float)
