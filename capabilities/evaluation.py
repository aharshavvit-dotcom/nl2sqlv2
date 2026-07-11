from __future__ import annotations

from typing import Iterable

from .taxonomy import ALL_CAPABILITIES


class CapabilityEvaluator:
    def evaluate(
        self,
        gold_labels: Iterable[Iterable[str]],
        predicted_scores: Iterable[dict[str, float] | Iterable[str]],
        *,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, object]:
        threshold_map = thresholds or {}
        gold_sets = [set(labels) for labels in gold_labels]
        pred_inputs = list(predicted_scores)
        pred_sets = [
            self._predicted_set(item, threshold_map)
            for item in pred_inputs
        ]
        per_capability: dict[str, dict[str, float | int]] = {}
        total_tp = total_fp = total_fn = 0
        for capability in ALL_CAPABILITIES:
            name = capability.value
            tp = sum(1 for gold, pred in zip(gold_sets, pred_sets) if name in gold and name in pred)
            fp = sum(1 for gold, pred in zip(gold_sets, pred_sets) if name not in gold and name in pred)
            fn = sum(1 for gold, pred in zip(gold_sets, pred_sets) if name in gold and name not in pred)
            support = sum(1 for gold in gold_sets if name in gold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            per_capability[name] = {
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "support": support,
                "average_precision": self._average_precision(name, gold_sets, pred_inputs),
            }
        micro_precision = _safe_div(total_tp, total_tp + total_fp)
        micro_recall = _safe_div(total_tp, total_tp + total_fn)
        macro_precision = sum(float(item["precision"]) for item in per_capability.values()) / len(per_capability)
        macro_recall = sum(float(item["recall"]) for item in per_capability.values()) / len(per_capability)
        macro_f1 = sum(float(item["f1"]) for item in per_capability.values()) / len(per_capability)
        exact_match = _safe_div(sum(1 for gold, pred in zip(gold_sets, pred_sets) if gold == pred), len(gold_sets))
        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": _f1(micro_precision, micro_recall),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "exact_multilabel_match": exact_match,
            "per_capability": per_capability,
        }

    @staticmethod
    def _predicted_set(prediction: dict[str, float] | Iterable[str], thresholds: dict[str, float]) -> set[str]:
        if isinstance(prediction, dict):
            return {
                name
                for name, score in prediction.items()
                if float(score) >= float(thresholds.get(name, 0.5))
            }
        return {str(item) for item in prediction}

    @staticmethod
    def _average_precision(
        capability: str,
        gold_sets: list[set[str]],
        predictions: list[dict[str, float] | Iterable[str]],
    ) -> float | None:
        if not any(capability in gold for gold in gold_sets):
            return None
        if not all(isinstance(item, dict) for item in predictions):
            return None
        scored = sorted(
            [
                (float(prediction.get(capability, 0.0)), capability in gold)
                for prediction, gold in zip(predictions, gold_sets)
                if isinstance(prediction, dict)
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        positives = sum(1 for _, is_positive in scored if is_positive)
        if positives == 0:
            return None
        hits = 0
        precision_sum = 0.0
        for index, (_, is_positive) in enumerate(scored, start=1):
            if not is_positive:
                continue
            hits += 1
            precision_sum += hits / index
        return precision_sum / positives


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
