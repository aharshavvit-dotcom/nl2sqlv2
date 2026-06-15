from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


SLOT_KEYS = [
    "intent",
    "base_table",
    "metric_column",
    "dimension_column",
    "date_column",
    "filter_column",
    "filter_operator",
]


class OptionAErrorAnalyzer:
    def analyze(self, prediction_rows: list[dict]) -> dict[str, Any]:
        by_intent: dict[str, Counter] = defaultdict(Counter)
        by_dataset: dict[str, Counter] = defaultdict(Counter)
        by_failure_type: Counter = Counter()
        slot_correct = Counter()
        slot_total = Counter()
        failures = []

        for row in prediction_rows:
            gold = row.get("gold") or {}
            pred = row.get("prediction") or row.get("decoded_prediction") or {}
            intent = str(gold.get("intent") or gold.get("template_id") or row.get("intent") or "unknown")
            dataset = str(row.get("dataset_name") or row.get("dataset") or "unknown")
            by_intent[intent]["total"] += 1
            by_dataset[dataset]["total"] += 1

            row_failures = _failure_types(row, gold, pred)
            if not row_failures:
                by_intent[intent]["ok"] += 1
                by_dataset[dataset]["ok"] += 1
            for failure in row_failures:
                by_failure_type[failure] += 1
                by_intent[intent][failure] += 1
                by_dataset[dataset][failure] += 1

            for slot in SLOT_KEYS:
                if slot not in gold:
                    continue
                slot_total[slot] += 1
                if _same_slot(gold.get(slot), pred.get(slot)):
                    slot_correct[slot] += 1

            if row_failures and len(failures) < 25:
                failures.append(
                    {
                        "id": row.get("id") or row.get("example_id"),
                        "question": row.get("question"),
                        "failures": row_failures,
                        "gold": gold,
                        "prediction": pred,
                    }
                )

        report = {
            "total": len(prediction_rows),
            "by_intent": _counter_map(by_intent),
            "by_dataset": _counter_map(by_dataset),
            "by_failure_type": dict(by_failure_type),
            "slot_accuracy": {
                slot: slot_correct[slot] / max(slot_total[slot], 1)
                for slot in SLOT_KEYS
            },
            "top_failure_examples": failures,
            "recommendations": _recommendations(by_failure_type),
        }
        return report


def _failure_types(row: dict[str, Any], gold: dict[str, Any], pred: dict[str, Any]) -> list[str]:
    failures = []
    if gold and pred:
        if gold.get("intent") != pred.get("intent"):
            failures.append("wrong intent")
        for slot in ["base_table", "metric_column", "dimension_column", "date_column", "filter_column"]:
            if slot in gold and not _same_slot(gold.get(slot), pred.get(slot)):
                failures.append(f"wrong {slot.replace('_', ' ')}")
    ir_validation = row.get("ir_validation") or {}
    if not ir_validation.get("is_valid", True):
        failures.append("invalid QueryIR")
    sql_validation = row.get("sql_validation") or row.get("validation") or {}
    if not sql_validation.get("is_valid", sql_validation.get("ok", True)):
        failures.append("invalid SQL")
        for issue in sql_validation.get("issues") or []:
            failures.append(f"SQL validation failure: {issue}")
    return list(dict.fromkeys(failures))


def _same_slot(left: Any, right: Any) -> bool:
    return _slot_key(left) == _slot_key(right)


def _slot_key(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("table") and value.get("column"):
            return (value.get("table"), value.get("column"))
        return tuple(sorted(value.items()))
    return value


def _counter_map(values: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {key: dict(counter) for key, counter in sorted(values.items())}


def _recommendations(failures: Counter) -> list[str]:
    recommendations = []
    if failures.get("wrong metric column", 0):
        recommendations.append("Improve metric candidate masks and sales/revenue synonym coverage.")
    if failures.get("wrong dimension column", 0):
        recommendations.append("Add hard negatives for name/status/region/category dimension slots.")
    if failures.get("wrong intent", 0):
        recommendations.append("Balance curriculum examples by intent before BIRD fine-tuning.")
    if failures.get("invalid SQL", 0):
        recommendations.append("Inspect SQL validation failures and add schema-linker constraints for the failing columns.")
    if not recommendations:
        recommendations.append("No dominant failure pattern detected; review low-confidence examples next.")
    return recommendations

