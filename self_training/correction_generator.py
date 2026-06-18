"""Correction Example Generator — creates (wrong → gold) training pairs.

These correction examples are appended to the training set with a higher
weight so the model specifically learns from its mistakes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .error_classifier import ErrorCategory, ErrorClassification


# Map from error category to correction type
_CORRECTION_TYPE: dict[ErrorCategory, str] = {
    ErrorCategory.WRONG_INTENT: "intent_correction",
    ErrorCategory.WRONG_BASE_TABLE: "table_correction",
    ErrorCategory.WRONG_METRIC: "column_correction",
    ErrorCategory.WRONG_DIMENSION: "column_correction",
    ErrorCategory.WRONG_FILTER: "filter_correction",
    ErrorCategory.WRONG_DATE_FILTER: "filter_correction",
    ErrorCategory.WRONG_JOIN: "join_correction",
    ErrorCategory.UNNECESSARY_JOIN: "join_correction",
    ErrorCategory.MISSING_JOIN: "join_correction",
    ErrorCategory.WRONG_ORDER: "order_correction",
    ErrorCategory.WRONG_LIMIT: "limit_correction",
    ErrorCategory.SQL_VALIDATION_FAILURE: "validation_fix",
    ErrorCategory.IR_VALIDATION_FAILURE: "validation_fix",
    ErrorCategory.CONVERSION_FAILURE: "conversion_fix",
}


class CorrectionExampleGenerator:
    """Generates correction training examples from prediction errors."""

    def __init__(self, correction_weight: float = 2.0):
        self.correction_weight = correction_weight

    def generate(
        self,
        classifications: list[ErrorClassification],
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create correction examples for each classified error.

        Each example pairs the wrong prediction with the gold QueryIR,
        tagged with the error categories and a correction weight.
        """

        example_map = {str(ex.get("example_id", idx)): ex for idx, ex in enumerate(examples)}
        corrections: list[dict[str, Any]] = []

        for ec in classifications:
            if not ec.categories:
                continue

            example = example_map.get(ec.example_id, {})
            gold_ir = example.get("query_ir") or example.get("gold_query_ir") or {}
            pred_ir = example.get("predicted_query_ir") or {}

            if not gold_ir:
                continue

            # Determine correction type from the primary error category
            primary_category = ec.categories[0]
            correction_type = _CORRECTION_TYPE.get(primary_category, "general_correction")

            correction = {
                "example_id": f"{ec.example_id}_correction",
                "original_example_id": ec.example_id,
                "question": example.get("question", ""),
                "dataset_name": example.get("dataset_name", ""),
                "db_id": example.get("db_id", ""),
                "split": example.get("split", "train"),
                "schema": example.get("schema") or example.get("serialized_schema"),
                "serialized_schema": example.get("serialized_schema"),
                "source_sql": example.get("source_sql", ""),
                "query_ir": deepcopy(gold_ir),
                "wrong_prediction": deepcopy(pred_ir) if pred_ir else None,
                "error_categories": [c.value for c in ec.categories],
                "correction_type": correction_type,
                "severity": ec.severity,
                "suggested_fix_type": ec.suggested_fix_type,
                "intent": gold_ir.get("intent"),
                "template_id": gold_ir.get("template_id"),
                "metadata": {
                    "source": "correction",
                    "correction_weight": self.correction_weight,
                    "original_match_score": None,
                    "error_details": ec.details,
                },
            }

            # Copy over rendered SQL and validation info if present
            for key in ("rendered_sql", "ir_validation", "sql_validation", "sql_features", "complexity"):
                if key in example:
                    correction[key] = example[key]

            corrections.append(correction)

        return corrections

    def generate_augmented_training_set(
        self,
        original_train: list[dict[str, Any]],
        corrections: list[dict[str, Any]],
        hard_negatives: list[dict[str, Any]],
        hard_negative_weight: float = 1.5,
    ) -> list[dict[str, Any]]:
        """Merge original training data with corrections and hard negatives.

        Original examples get ``metadata.source = "original"``, corrections
        get ``"correction"`` with their weight, and hard negatives get
        ``"hard_negative"`` with their weight.
        """

        augmented: list[dict[str, Any]] = []

        # Original training examples
        for row in original_train:
            row = deepcopy(row)
            metadata = row.get("metadata") or {}
            metadata["source"] = metadata.get("source", "original")
            metadata.setdefault("correction_weight", 1.0)
            row["metadata"] = metadata
            augmented.append(row)

        # Correction examples (added with higher weight)
        for corr in corrections:
            corr = deepcopy(corr)
            metadata = corr.get("metadata") or {}
            metadata["source"] = "correction"
            metadata.setdefault("correction_weight", self.correction_weight)
            corr["metadata"] = metadata
            augmented.append(corr)

        # Hard negatives (as negative examples with their own weight)
        for neg in hard_negatives:
            neg = deepcopy(neg)
            # Convert hard negative into a training example with the gold as target
            training_row = {
                "example_id": neg.get("negative_id") or neg.get("example_id", ""),
                "question": neg.get("question", ""),
                "dataset_name": neg.get("dataset_name", ""),
                "db_id": neg.get("db_id", ""),
                "query_ir": neg.get("gold_query_ir", {}),
                "negative_query_ir": neg.get("negative_query_ir", {}),
                "negative_type": neg.get("negative_type", ""),
                "metadata": {
                    "source": "hard_negative",
                    "correction_weight": hard_negative_weight,
                    "negative_source": neg.get("source", "prediction_error"),
                },
            }
            augmented.append(training_row)

        return augmented
