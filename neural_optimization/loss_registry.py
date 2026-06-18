"""Loss function registry.

Maps human-readable loss names to PyTorch loss function constructors.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.nn import functional as F


def cross_entropy_loss(ignore_index: int = -1) -> Callable:
    """Standard cross-entropy with configurable ignore index."""
    return torch.nn.CrossEntropyLoss(ignore_index=ignore_index)


def masked_cross_entropy_fn(logits, targets, mask=None, ignore_index: int = -1):
    """Masked cross-entropy used for pointer heads.

    Re-exports the function from ``neural_ir.loss_utils`` so that the
    optimized trainer can use the same loss without a circular import.
    """
    effective_targets = targets
    if mask is not None:
        mask = mask.to(logits.device).bool()
        logits = logits.masked_fill(~mask, -1e9)
        valid_target = targets.ne(ignore_index)
        target_ok = torch.zeros_like(valid_target, dtype=torch.bool)
        in_range = targets.ge(0) & targets.lt(mask.size(-1))
        safe_targets = targets.clamp(0, max(mask.size(-1) - 1, 0))
        gathered = mask.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        target_ok[in_range] = gathered[in_range]
        effective_targets = torch.where(valid_target & target_ok, targets, torch.full_like(targets, ignore_index))
    if not effective_targets.ne(ignore_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, effective_targets, ignore_index=ignore_index)


def margin_ranking_loss(gold_scores, negative_scores, margin: float = 0.2):
    """Margin ranking loss for hard-negative training."""
    if gold_scores.numel() == 0 or negative_scores.numel() == 0:
        return (gold_scores.sum() + negative_scores.sum()) * 0.0
    target = torch.ones_like(gold_scores)
    return F.margin_ranking_loss(gold_scores, negative_scores, target, margin=margin)


LOSS_REGISTRY: dict[str, Any] = {
    "cross_entropy": cross_entropy_loss,
    "masked_cross_entropy": masked_cross_entropy_fn,
    "margin_ranking": margin_ranking_loss,
}


def get_loss_fn(name: str) -> Any:
    """Look up a loss function by name."""
    if name not in LOSS_REGISTRY:
        supported = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Supported: {supported}")
    return LOSS_REGISTRY[name]
