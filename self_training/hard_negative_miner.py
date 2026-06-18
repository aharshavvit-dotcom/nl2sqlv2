from __future__ import annotations

from collections import Counter
from typing import Any

from .error_classifier import ErrorClassifier
from .gold_comparator import GoldComparator


class HardNegativeMiner:
    def mine(self, prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
        comparator = GoldComparator()
        classifier = ErrorClassifier()
        negatives = []
        error_counts: Counter[str] = Counter()
        for row in prediction_rows:
            gold_ir = row.get("gold_query_ir") or row.get("query_ir") or {}
            pred_ir = row.get("predicted_query_ir") or {}
            if not gold_ir or not pred_ir or pred_ir == gold_ir:
                continue
            comparison = comparator.compare_query_ir(pred_ir, gold_ir, example_id=str(row.get("example_id") or ""))
            classification = classifier.classify(comparison, row)
            categories = [category.value for category in classification.categories] or ["unknown_error"]
            for category in categories:
                error_counts[category] += 1
            negatives.append(
                {
                    "example_id": row.get("example_id"),
                    "negative_id": f"{row.get('example_id')}_mined_negative",
                    "question": row.get("question"),
                    "dataset_name": row.get("dataset_name"),
                    "db_id": row.get("db_id"),
                    "gold_query_ir": gold_ir,
                    "negative_query_ir": pred_ir,
                    "negative_type": categories[0],
                    "error_categories": categories,
                    "source": "validation_model_mistake",
                }
            )
        return {
            "mined_hard_negatives": negatives,
            "error_summary": {"total_errors": len(negatives), "by_category": dict(error_counts)},
        }
