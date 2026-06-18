"""Tests for neural_optimization.scheduler_factory."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from neural_optimization.scheduler_factory import build_scheduler


def _dummy_optimizer():
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=0.001)


class TestSchedulerFactory:
    def test_none_returns_none(self):
        assert build_scheduler(_dummy_optimizer(), {"name": "none"}) is None

    def test_null_returns_none(self):
        assert build_scheduler(_dummy_optimizer(), {"name": "null"}) is None

    def test_empty_returns_none(self):
        assert build_scheduler(_dummy_optimizer(), {}) is None

    def test_reduce_on_plateau(self):
        sch = build_scheduler(_dummy_optimizer(), {
            "name": "reduce_on_plateau", "factor": 0.5, "patience": 2,
        })
        assert isinstance(sch, ReduceLROnPlateau)

    def test_step_lr(self):
        sch = build_scheduler(_dummy_optimizer(), {
            "name": "step_lr", "step_size": 5, "factor": 0.5,
        })
        assert isinstance(sch, StepLR)

    def test_cosine_annealing(self):
        sch = build_scheduler(_dummy_optimizer(), {
            "name": "cosine_annealing", "t_max": 10,
        })
        assert isinstance(sch, CosineAnnealingLR)

    def test_one_cycle(self):
        sch = build_scheduler(_dummy_optimizer(), {
            "name": "one_cycle", "total_steps": 100,
        })
        assert sch is not None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(_dummy_optimizer(), {"name": "invalid"})
