"""Optimizer factory.

Builds a PyTorch optimizer from a config dict.  Default is AdamW.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def build_optimizer(
    model_parameters: nn.Module | Iterable[nn.Parameter] | Iterable[tuple[str, nn.Parameter]],
    config: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Build a PyTorch optimizer from *config*.

    Parameters
    ----------
    model_parameters:
        ``model.parameters()`` iterable.
    config:
        Dict with at least ``name``.  Also supports ``learning_rate``,
        ``weight_decay``, ``momentum``, ``nesterov``.

    Raises
    ------
    ValueError
        If *name* is not a recognised optimizer.
    """
    config = config or {}
    name = str(config.get("name", "adamw")).strip().lower()
    lr = float(config.get("learning_rate", 0.0007))
    wd = float(config.get("weight_decay", 0.0))
    momentum = float(config.get("momentum", 0.9))
    nesterov = bool(config.get("nesterov", False))

    params = _parameter_groups(model_parameters, wd, config)

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)

    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)

    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)

    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    if name == "nadam":
        if not hasattr(torch.optim, "NAdam"):
            raise ValueError("NAdam requires PyTorch >= 1.10. Please upgrade or choose another optimizer.")
        return torch.optim.NAdam(params, lr=lr, weight_decay=wd)

    supported = "sgd, momentum, nesterov, rmsprop, adam, adamw, nadam"
    raise ValueError(f"Unknown optimizer '{name}'. Supported: {supported}")


_POINTER_HEAD_NAMES = (
    "table_pointer",
    "metric_pointer",
    "dimension_pointer",
    "date_pointer",
    "filter_pointer",
)


def _parameter_groups(
    model_parameters: nn.Module | Iterable[nn.Parameter] | Iterable[tuple[str, nn.Parameter]],
    weight_decay: float,
    config: dict[str, Any],
) -> list[Any]:
    """Return optimizer groups, separating schema-pointer heads when names are available."""
    pointer_decay = float(config.get("pointer_head_weight_decay", weight_decay))
    if isinstance(model_parameters, nn.Module):
        named = list(model_parameters.named_parameters())
    else:
        values = list(model_parameters)
        if values and isinstance(values[0], tuple):
            named = values  # type: ignore[assignment]
        else:
            return values

    pointer_params = [
        parameter
        for name, parameter in named
        if parameter.requires_grad and any(token in name for token in _POINTER_HEAD_NAMES)
    ]
    non_pointer_params = [
        parameter
        for name, parameter in named
        if parameter.requires_grad and not any(token in name for token in _POINTER_HEAD_NAMES)
    ]
    groups: list[dict[str, Any]] = []
    if non_pointer_params:
        groups.append({
            "params": non_pointer_params,
            "weight_decay": weight_decay,
            "group_name": "non_pointer_params",
        })
    if pointer_params:
        groups.append({
            "params": pointer_params,
            "weight_decay": pointer_decay,
            "group_name": "pointer_head_params",
        })
    return groups
