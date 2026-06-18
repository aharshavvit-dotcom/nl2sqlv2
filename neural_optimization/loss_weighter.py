"""Per-head loss weighting for multi-task training.

Combines individual head losses into a single scalar using configurable
per-head weights.
"""

from __future__ import annotations

from typing import Any

import torch


class MultiTaskLossWeighter:
    """Combines per-head losses using configurable weights.

    Parameters
    ----------
    weights:
        Mapping of head name → weight.  Heads not listed have weight 1.0.
        A weight of 0 disables that head.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights: dict[str, float] = dict(weights or {})

    def combine(self, losses: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Combine individual head losses into a weighted total.

        Parameters
        ----------
        losses:
            Mapping head_name → scalar loss tensor.

        Returns
        -------
        dict with ``total_loss`` (tensor), ``weighted_losses`` (dict),
        and ``raw_losses`` (dict of floats).
        """
        weighted: dict[str, torch.Tensor] = {}
        raw: dict[str, float] = {}
        parts: list[torch.Tensor] = []

        for name, loss in losses.items():
            if loss is None:
                continue
            weight = self.weights.get(name, 1.0)
            raw[name] = float(loss.item())
            wl = loss * weight
            weighted[name] = wl
            parts.append(wl)

        if parts:
            total = torch.stack(parts).sum()
        else:
            # Return a zero-gradient tensor from one of the input losses
            any_loss = next(iter(losses.values()), None)
            total = any_loss.sum() * 0.0 if any_loss is not None else torch.tensor(0.0)

        return {
            "total_loss": total,
            "weighted_losses": {k: float(v.item()) for k, v in weighted.items()},
            "raw_losses": raw,
        }
