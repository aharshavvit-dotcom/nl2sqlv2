"""Tests for neural_optimization.ffn_blocks."""

from __future__ import annotations

import torch
from neural_optimization.ffn_blocks import FeedForwardBlock


class TestFeedForwardBlock:
    def test_forward_pass(self):
        block = FeedForwardBlock(input_dim=64, hidden_dim=128, output_dim=32)
        x = torch.randn(4, 64)
        out = block(x)
        assert out.shape == (4, 32)

    def test_output_shape_default(self):
        """When output_dim is None, output_dim defaults to input_dim."""
        block = FeedForwardBlock(input_dim=64, hidden_dim=128)
        x = torch.randn(2, 64)
        out = block(x)
        assert out.shape == (2, 64)

    def test_residual_works(self):
        block = FeedForwardBlock(input_dim=64, hidden_dim=128, output_dim=64, residual=True)
        x = torch.randn(2, 64)
        out = block(x)
        assert out.shape == (2, 64)
        # Residual should make output differ from a non-residual block
        block_no_res = FeedForwardBlock(input_dim=64, hidden_dim=128, output_dim=64, residual=False)
        block_no_res.load_state_dict(block.state_dict())
        out_no_res = block_no_res(x)
        # The outputs should differ because residual adds x
        assert not torch.allclose(out, out_no_res)

    def test_residual_disabled_on_dim_mismatch(self):
        """Residual should be disabled when input_dim != output_dim."""
        block = FeedForwardBlock(input_dim=64, hidden_dim=128, output_dim=32, residual=True)
        assert block.residual is False

    def test_dropout_configurable(self):
        block = FeedForwardBlock(input_dim=32, hidden_dim=64, dropout=0.5)
        # Check that a Dropout layer exists in the net
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in block.net.modules())
        assert has_dropout

    def test_layer_norm_included(self):
        block = FeedForwardBlock(input_dim=32, hidden_dim=64, layer_norm=True)
        has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in block.net.modules())
        assert has_ln

    def test_layer_norm_excluded(self):
        block = FeedForwardBlock(input_dim=32, hidden_dim=64, layer_norm=False)
        has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in block.net.modules())
        assert not has_ln

    def test_activation_configurable(self):
        block = FeedForwardBlock(input_dim=32, hidden_dim=64, activation="relu")
        has_relu = any(isinstance(m, torch.nn.ReLU) for m in block.net.modules())
        assert has_relu
