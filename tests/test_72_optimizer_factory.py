"""Tests for neural_optimization.optimizer_factory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_optimization.optimizer_factory import build_optimizer


def _dummy_params():
    return nn.Linear(4, 2).parameters()


class TestOptimizerFactory:
    def test_adamw(self):
        opt = build_optimizer(_dummy_params(), {"name": "adamw", "learning_rate": 0.001})
        assert isinstance(opt, torch.optim.AdamW)

    def test_adam(self):
        opt = build_optimizer(_dummy_params(), {"name": "adam"})
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self):
        opt = build_optimizer(_dummy_params(), {"name": "sgd"})
        assert isinstance(opt, torch.optim.SGD)

    def test_momentum(self):
        opt = build_optimizer(_dummy_params(), {"name": "momentum", "momentum": 0.9})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["momentum"] == 0.9

    def test_nesterov(self):
        opt = build_optimizer(_dummy_params(), {"name": "nesterov"})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults.get("nesterov") is True

    def test_rmsprop(self):
        opt = build_optimizer(_dummy_params(), {"name": "rmsprop"})
        assert isinstance(opt, torch.optim.RMSprop)

    def test_nadam(self):
        if not hasattr(torch.optim, "NAdam"):
            pytest.skip("NAdam not available")
        opt = build_optimizer(_dummy_params(), {"name": "nadam"})
        assert isinstance(opt, torch.optim.NAdam)

    def test_default_is_adamw(self):
        opt = build_optimizer(_dummy_params(), {})
        assert isinstance(opt, torch.optim.AdamW)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(_dummy_params(), {"name": "invalid"})
