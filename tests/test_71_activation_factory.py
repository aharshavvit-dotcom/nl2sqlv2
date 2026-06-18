"""Tests for neural_optimization.activation_factory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_optimization.activation_factory import get_activation


class TestActivationFactory:
    def test_relu(self):
        act = get_activation("relu")
        assert isinstance(act, nn.ReLU)

    def test_gelu(self):
        act = get_activation("gelu")
        assert isinstance(act, nn.GELU)

    def test_leaky_relu(self):
        act = get_activation("leaky_relu")
        assert isinstance(act, nn.LeakyReLU)

    def test_tanh(self):
        act = get_activation("tanh")
        assert isinstance(act, nn.Tanh)

    def test_sigmoid(self):
        act = get_activation("sigmoid")
        assert isinstance(act, nn.Sigmoid)

    def test_identity(self):
        act = get_activation("identity")
        x = torch.randn(2, 3)
        assert torch.equal(act(x), x)

    def test_default_is_gelu(self):
        act = get_activation(None)
        assert isinstance(act, nn.GELU)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("invalid_activation")

    def test_case_insensitive(self):
        assert isinstance(get_activation("GELU"), nn.GELU)
        assert isinstance(get_activation("ReLU"), nn.ReLU)
