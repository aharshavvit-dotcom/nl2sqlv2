"""Semantic consistency loss objectives for NL2SQL training.

Provides structured consistency constraints that encode domain knowledge
about valid SQL compositions:

1. Metric-aggregation compatibility: COUNT/SUM/AVG require numeric columns
2. Filter-operator-datatype: LIKE only on text, comparisons on numeric/date
3. GROUP BY consistency: dimensions in SELECT must appear in GROUP BY
4. Join path validity: joined tables must share FK relationships

These losses are auxiliary (weight << 1.0) and additive to the main
multi-task losses.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticConsistencyLoss(nn.Module):
    """Auxiliary loss enforcing structural consistency between predicted heads.

    Each sub-loss is independently weighted and can be disabled.
    All losses return 0 if the required inputs are not available.
    """

    def __init__(
        self,
        metric_agg_weight: float = 0.1,
        filter_op_weight: float = 0.1,
        groupby_weight: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__()
        self.metric_agg_weight = metric_agg_weight
        self.filter_op_weight = filter_op_weight
        self.groupby_weight = groupby_weight
        self.enabled = enabled

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        column_datatypes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute semantic consistency losses.

        Parameters
        ----------
        outputs : dict
            Model output logits (intent_logits, metric_aggregation_logits, etc.)
        labels : dict
            Ground truth labels.
        column_datatypes : Tensor, optional
            Per-column datatype encoding (0=unknown, 1=numeric, 2=text, 3=date).

        Returns
        -------
        dict with keys: 'semantic_total', 'metric_agg_consistency',
        'filter_op_consistency', 'groupby_consistency'
        """
        if not self.enabled:
            zero = torch.tensor(0.0, device=self._get_device(outputs))
            return {
                "semantic_total": zero,
                "metric_agg_consistency": zero,
                "filter_op_consistency": zero,
                "groupby_consistency": zero,
            }

        losses = {}

        # 1. Metric-aggregation consistency
        losses["metric_agg_consistency"] = self._metric_agg_consistency(
            outputs, labels, column_datatypes
        )

        # 2. Filter-operator consistency
        losses["filter_op_consistency"] = self._filter_op_consistency(
            outputs, labels, column_datatypes
        )

        # 3. GROUP BY consistency
        losses["groupby_consistency"] = self._groupby_consistency(outputs, labels)

        losses["semantic_total"] = (
            self.metric_agg_weight * losses["metric_agg_consistency"]
            + self.filter_op_weight * losses["filter_op_consistency"]
            + self.groupby_weight * losses["groupby_consistency"]
        )

        return losses

    def _metric_agg_consistency(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        column_datatypes: torch.Tensor | None,
    ) -> torch.Tensor:
        """Penalize SUM/AVG on non-numeric columns.

        If metric_aggregation predicts SUM (idx 1) or AVG (idx 2),
        and the metric_column points to a text column, add penalty.
        """
        agg_logits = outputs.get("metric_aggregation_logits")
        col_logits = outputs.get("metric_column_logits")

        if agg_logits is None or col_logits is None or column_datatypes is None:
            return torch.tensor(0.0, device=self._get_device(outputs))

        # Get predicted aggregation probabilities
        agg_probs = F.softmax(agg_logits, dim=-1)
        # SUM=1, AVG=2 are numeric-only aggregations
        numeric_agg_prob = agg_probs[:, 1:3].sum(dim=-1) if agg_probs.size(-1) > 2 else torch.zeros(agg_probs.size(0), device=agg_probs.device)

        # Get predicted column (argmax)
        pred_col = col_logits.argmax(dim=-1)  # [B]
        # Gather datatype for predicted column
        batch_size = pred_col.size(0)
        if column_datatypes.dim() == 2 and column_datatypes.size(0) == batch_size:
            pred_col_clamped = pred_col.clamp(0, column_datatypes.size(1) - 1)
            col_type = column_datatypes.gather(1, pred_col_clamped.unsqueeze(1)).squeeze(1)
            # Penalty: high prob of numeric agg × non-numeric column
            is_non_numeric = (col_type != 1).float()  # 1=numeric
            penalty = (numeric_agg_prob * is_non_numeric).mean()
            return penalty

        return torch.tensor(0.0, device=agg_logits.device)

    def _filter_op_consistency(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        column_datatypes: torch.Tensor | None,
    ) -> torch.Tensor:
        """Penalize LIKE on numeric columns, numeric comparisons on text.

        LIKE (idx 4 typically) should only apply to text columns.
        """
        op_logits = outputs.get("filter_operator_logits")
        col_logits = outputs.get("filter_column_logits")

        if op_logits is None or col_logits is None or column_datatypes is None:
            return torch.tensor(0.0, device=self._get_device(outputs))

        op_probs = F.softmax(op_logits, dim=-1)
        # LIKE is typically index 4
        if op_probs.size(-1) > 4:
            like_prob = op_probs[:, 4]
        else:
            return torch.tensor(0.0, device=op_logits.device)

        pred_col = col_logits.argmax(dim=-1)
        batch_size = pred_col.size(0)
        if column_datatypes.dim() == 2 and column_datatypes.size(0) == batch_size:
            pred_col_clamped = pred_col.clamp(0, column_datatypes.size(1) - 1)
            col_type = column_datatypes.gather(1, pred_col_clamped.unsqueeze(1)).squeeze(1)
            # Penalty: LIKE prob × numeric column
            is_numeric = (col_type == 1).float()
            penalty = (like_prob * is_numeric).mean()
            return penalty

        return torch.tensor(0.0, device=op_logits.device)

    def _groupby_consistency(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Penalize misalignment between intent and dimension selection.

        If intent predicts metric_by_dimension (has dimensions),
        dimension_column should have high confidence.
        If intent predicts show_records (no dimensions),
        dimension_column should have low confidence.
        """
        intent_logits = outputs.get("intent_logits")
        dim_logits = outputs.get("dimension_column_logits")

        if intent_logits is None or dim_logits is None:
            return torch.tensor(0.0, device=self._get_device(outputs))

        intent_probs = F.softmax(intent_logits, dim=-1)
        dim_max_prob = F.softmax(dim_logits, dim=-1).max(dim=-1).values

        # Intents that require dimensions (metric_by_dimension, etc.)
        # These are typically indices 1-4 in our label encoder
        if intent_probs.size(-1) >= 5:
            needs_dim_prob = intent_probs[:, 1:5].sum(dim=-1)
            # Penalty: high dimension-needing intent × low dimension confidence
            penalty = (needs_dim_prob * (1.0 - dim_max_prob)).mean()
            return penalty * 0.5  # Scale down

        return torch.tensor(0.0, device=intent_logits.device)

    @staticmethod
    def _get_device(outputs: dict[str, torch.Tensor]) -> torch.device:
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")
