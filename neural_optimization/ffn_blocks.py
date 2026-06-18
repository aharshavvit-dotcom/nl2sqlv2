"""Reusable feed-forward neural network block.

Used to add non-linear capacity in classification / pointer heads
of the Neural QueryIR Model.
"""

from __future__ import annotations

from torch import nn

from .activation_factory import get_activation


class FeedForwardBlock(nn.Module):
    """Two-layer feed-forward block with optional LayerNorm and residual.

    Internal structure::

        Linear(input_dim → hidden_dim)
        [LayerNorm]
        Activation
        Dropout
        Linear(hidden_dim → output_dim)
        [Residual when input_dim == output_dim]

    Parameters
    ----------
    input_dim:
        Input feature size.
    hidden_dim:
        Hidden layer size.
    output_dim:
        Output size.  Defaults to *input_dim* when ``None``.
    activation:
        Name passed to :func:`get_activation`.
    dropout:
        Dropout probability after activation.
    layer_norm:
        Whether to apply ``LayerNorm`` after the first linear layer.
    residual:
        Whether to add a skip connection when *input_dim* == *output_dim*.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.25,
        layer_norm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.residual = residual and (input_dim == output_dim)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = out + x
        return out
