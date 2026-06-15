from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_cross_entropy(logits, targets, mask=None, ignore_index: int = -1):
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


def accuracy_from_logits(logits, targets, mask=None, ignore_index: int = -1) -> tuple[int, int]:
    if mask is not None:
        logits = logits.masked_fill(~mask.to(logits.device).bool(), -1e9)
    valid = targets.ne(ignore_index)
    total = int(valid.sum().item())
    if total == 0:
        return 0, 0
    pred = logits.argmax(dim=-1)
    correct = int(pred.eq(targets).logical_and(valid).sum().item())
    return correct, total


def slot_accuracy(metrics: dict) -> float:
    values = [
        float(value)
        for key, value in metrics.items()
        if key.endswith("_accuracy") and key not in {"intent_accuracy", "template_accuracy"}
    ]
    return sum(values) / max(len(values), 1)


def margin_ranking_slot_loss(gold_scores, negative_scores, margin: float = 0.2):
    if gold_scores.numel() == 0 or negative_scores.numel() == 0:
        return (gold_scores.sum() + negative_scores.sum()) * 0.0
    target = torch.ones_like(gold_scores)
    return F.margin_ranking_loss(gold_scores, negative_scores, target, margin=margin)
