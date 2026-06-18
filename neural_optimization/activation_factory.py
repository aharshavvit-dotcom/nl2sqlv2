"""Activation function factory.

Returns ``nn.Module`` instances for use in FFN heads and other layers.
Default is GELU; fallback is ReLU.
"""

from __future__ import annotations

from torch import nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class _Identity(nn.Module):
    """No-op activation (identity function)."""

    def forward(self, x):
        return x


def get_activation(name: str | None = None) -> nn.Module:
    """Return an ``nn.Module`` activation by name.

    Parameters
    ----------
    name:
        One of ``relu``, ``gelu``, ``leaky_relu``, ``tanh``, ``sigmoid``,
        ``identity``.  ``None`` or empty string default to ``gelu``.

    Raises
    ------
    ValueError
        If *name* is not a recognised activation.
    """
    name = (name or "gelu").strip().lower()
    if name == "identity":
        return _Identity()
    if name not in _ACTIVATIONS:
        supported = ", ".join(sorted([*_ACTIVATIONS, "identity"]))
        raise ValueError(f"Unknown activation '{name}'. Supported: {supported}")
    return _ACTIVATIONS[name]()
