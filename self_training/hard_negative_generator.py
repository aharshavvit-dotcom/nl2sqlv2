"""Hard Negative Generator — creates targeted hard negatives from prediction errors.

Unlike the synthetic ``HardNegativeCorpusBuilder`` in ``dataset_training``,
this module generates hard negatives from *actual* wrong predictions so the
model learns to avoid its own mistakes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .error_classifier import ErrorCategory, ErrorClassification


class PredictionHardNegativeGenerator:
    """Generates hard negatives from actual prediction errors."""

    def generate_from_errors(
        self,
        classifications: list[ErrorClassification],
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create hard negatives from actual wrong predictions.

        Each hard negative pairs the gold QueryIR with the model's wrong
        prediction so the model learns to distinguish them.
        """

        example_map = {str(ex.get("example_id", idx)): ex for idx, ex in enumerate(examples)}
        negatives: list[dict[str, Any]] = []

        for ec in classifications:
            example = example_map.get(ec.example_id, {})
            gold_ir = example.get("query_ir") or example.get("gold_query_ir") or {}
            pred_ir = example.get("predicted_query_ir") or {}

            if not gold_ir or not pred_ir:
                continue

            for cat in ec.categories:
                neg = self._negative_for_category(cat, example, gold_ir, pred_ir)
                if neg is not None:
                    negatives.append(neg)

        return negatives

    def generate_contrastive_pairs(
        self,
        classifications: list[ErrorClassification],
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create (gold, predicted_wrong) contrastive pairs for margin loss.

        Each pair is suitable for margin-ranking training: the model should
        score the gold higher than the wrong prediction.
        """

        example_map = {str(ex.get("example_id", idx)): ex for idx, ex in enumerate(examples)}
        pairs: list[dict[str, Any]] = []

        for ec in classifications:
            if not ec.categories:
                continue
            example = example_map.get(ec.example_id, {})
            gold_ir = example.get("query_ir") or example.get("gold_query_ir") or {}
            pred_ir = example.get("predicted_query_ir") or {}

            if not gold_ir or not pred_ir:
                continue

            pairs.append({
                "example_id": ec.example_id,
                "question": example.get("question", ""),
                "dataset_name": example.get("dataset_name", ""),
                "db_id": example.get("db_id", ""),
                "gold_query_ir": gold_ir,
                "predicted_query_ir": pred_ir,
                "error_categories": [c.value for c in ec.categories],
                "severity": ec.severity,
                "source": "contrastive_pair",
            })

        return pairs

    # ------------------------------------------------------------------
    # Category-specific negative generation
    # ------------------------------------------------------------------

    def _negative_for_category(
        self,
        category: ErrorCategory,
        example: dict[str, Any],
        gold_ir: dict[str, Any],
        pred_ir: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Create a single hard negative for a specific error category."""

        handler = _CATEGORY_HANDLERS.get(category)
        if handler is None:
            return None

        negative_ir = handler(pred_ir, gold_ir)
        if negative_ir is None:
            return None

        return {
            "example_id": example.get("example_id", ""),
            "negative_id": f"{example.get('example_id', '')}_neg_{category.value}",
            "question": example.get("question", ""),
            "dataset_name": example.get("dataset_name", ""),
            "db_id": example.get("db_id", ""),
            "gold_query_ir": gold_ir,
            "negative_query_ir": negative_ir,
            "negative_type": category.value,
            "source": "prediction_error",
        }


# ---------------------------------------------------------------------------
# Category-specific handlers
# ---------------------------------------------------------------------------

def _wrong_intent_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) intent as the negative."""
    if pred.get("intent") == gold.get("intent"):
        return None
    neg = deepcopy(gold)
    neg["intent"] = pred.get("intent")
    neg["template_id"] = pred.get("template_id") or pred.get("intent")
    return neg


def _wrong_base_table_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) base_table as the negative."""
    if pred.get("base_table") == gold.get("base_table"):
        return None
    neg = deepcopy(gold)
    neg["base_table"] = pred.get("base_table")
    neg["required_tables"] = [pred.get("base_table")]
    return neg


def _wrong_metric_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) metrics as the negative."""
    pred_metrics = pred.get("metrics") or []
    gold_metrics = gold.get("metrics") or []
    if not pred_metrics or pred_metrics == gold_metrics:
        return None
    neg = deepcopy(gold)
    neg["metrics"] = deepcopy(pred_metrics)
    return neg


def _wrong_dimension_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) dimensions as the negative."""
    pred_dims = pred.get("dimensions") or []
    gold_dims = gold.get("dimensions") or []
    if not pred_dims or pred_dims == gold_dims:
        return None
    neg = deepcopy(gold)
    neg["dimensions"] = deepcopy(pred_dims)
    return neg


def _wrong_filter_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) filters as the negative."""
    pred_filters = pred.get("filters") or []
    gold_filters = gold.get("filters") or []
    if not pred_filters or pred_filters == gold_filters:
        return None
    neg = deepcopy(gold)
    neg["filters"] = deepcopy(pred_filters)
    return neg


def _wrong_date_filter_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) date_filters as the negative."""
    pred_df = pred.get("date_filters") or []
    gold_df = gold.get("date_filters") or []
    if not pred_df or pred_df == gold_df:
        return None
    neg = deepcopy(gold)
    neg["date_filters"] = deepcopy(pred_df)
    return neg


def _unnecessary_join_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted IR with unnecessary joins as the negative."""
    pred_joins = pred.get("joins") or []
    gold_joins = gold.get("joins") or []
    if not pred_joins or gold_joins:
        return None
    neg = deepcopy(gold)
    neg["joins"] = deepcopy(pred_joins)
    neg["required_tables"] = list(dict.fromkeys(
        (gold.get("required_tables") or []) +
        [j.get("right_table") or j.get("left_table") for j in pred_joins if isinstance(j, dict)]
    ))
    return neg


def _wrong_join_negative(pred: dict, gold: dict) -> dict | None:
    """Use predicted (wrong) joins as the negative."""
    pred_joins = pred.get("joins") or []
    gold_joins = gold.get("joins") or []
    if not pred_joins or pred_joins == gold_joins:
        return None
    neg = deepcopy(gold)
    neg["joins"] = deepcopy(pred_joins)
    return neg


_CATEGORY_HANDLERS: dict[ErrorCategory, Any] = {
    ErrorCategory.WRONG_INTENT: _wrong_intent_negative,
    ErrorCategory.WRONG_BASE_TABLE: _wrong_base_table_negative,
    ErrorCategory.WRONG_METRIC: _wrong_metric_negative,
    ErrorCategory.WRONG_DIMENSION: _wrong_dimension_negative,
    ErrorCategory.WRONG_FILTER: _wrong_filter_negative,
    ErrorCategory.WRONG_DATE_FILTER: _wrong_date_filter_negative,
    ErrorCategory.UNNECESSARY_JOIN: _unnecessary_join_negative,
    ErrorCategory.WRONG_JOIN: _wrong_join_negative,
}
