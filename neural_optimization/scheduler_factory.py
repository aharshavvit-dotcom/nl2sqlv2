"""Learning-rate scheduler factory.

Builds a PyTorch LR scheduler from a config dict.  Default is
``ReduceLROnPlateau``.  A scheduler is optional — pass
``name: none`` or ``name: null`` to disable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any] | None = None,
    total_steps: int | None = None,
) -> Any | None:
    """Build a learning-rate scheduler.

    Parameters
    ----------
    optimizer:
        The optimizer whose LR will be adjusted.
    config:
        Dict with at least ``name``.  Scheduler-specific keys are forwarded.
    total_steps:
        Required only for ``one_cycle``.

    Returns
    -------
    scheduler or ``None`` if name is ``"none"`` / ``None``.
    """
    config = config or {}
    name = str(config.get("name") or "none").strip().lower()

    if name in ("none", "null", ""):
        return None

    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 2)),
            min_lr=float(config.get("min_lr", 1e-6)),
        )

    if name == "step_lr":
        return StepLR(
            optimizer,
            step_size=int(config.get("step_size", 5)),
            gamma=float(config.get("factor", 0.5)),
        )

    if name == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", config.get("T_max", 10))),
            eta_min=float(config.get("min_lr", 1e-6)),
        )

    if name == "one_cycle":
        steps = total_steps or int(config.get("total_steps", 100))
        return OneCycleLR(
            optimizer,
            max_lr=float(config.get("max_lr", optimizer.defaults.get("lr", 0.001))),
            total_steps=steps,
        )

    supported = "none, reduce_on_plateau, step_lr, cosine_annealing, one_cycle"
    raise ValueError(f"Unknown scheduler '{name}'. Supported: {supported}")
