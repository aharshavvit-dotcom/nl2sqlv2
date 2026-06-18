from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


class DatasetScaleEvaluator:
    def __init__(self, predictor: Any | None = None):
        self.predictor = predictor

    def evaluate_model(
        self,
        model_name: str,
        examples: list[dict[str, Any]],
        schema_mode: str = "gold",
        max_examples: int | None = None,
    ) -> dict[str, Any]:
        rows = examples[:max_examples] if max_examples is not None else examples
        failures: list[dict[str, Any]] = []
        metrics = Counter()
        totals = Counter()
        by_dataset = defaultdict(Counter)
        by_intent = defaultdict(Counter)
        by_complexity = defaultdict(Counter)
        by_database = defaultdict(Counter)

        for row in rows:
            gold = row.get("query_ir") or {}
            pred = self._predict(row, schema_mode=schema_mode)
            item_metrics = self._metrics(gold, pred, row)
            for key, value in item_metrics.items():
                totals[key] += 1
                if value:
                    metrics[key] += 1
            for bucket, name in [
                (by_dataset, row.get("dataset_name") or "unknown"),
                (by_intent, gold.get("intent") or row.get("intent") or "unknown"),
                (by_complexity, row.get("complexity") or "unknown"),
                (by_database, row.get("db_id") or "unknown"),
            ]:
                bucket[name]["total"] += 1
                for key, value in item_metrics.items():
                    if value:
                        bucket[name][key] += 1
            if not all(item_metrics.values()):
                failures.append({"example_id": row.get("example_id"), "question": row.get("question"), "metrics": item_metrics})

        summary = {f"{key}_rate": metrics[key] / totals[key] if totals[key] else 0.0 for key in totals}
        summary["total_examples"] = len(rows)
        summary["unnecessary_join_rate"] = 1.0 - summary.get("no_unnecessary_join_rate", 1.0)
        summary["wrong_table_rate"] = 1.0 - summary.get("base_table_accuracy_rate", 1.0)
        return {
            "model_name": model_name,
            "schema_mode": schema_mode,
            "summary": summary,
            "by_dataset": self._bucket_rates(by_dataset),
            "by_intent": self._bucket_rates(by_intent),
            "by_complexity": self._bucket_rates(by_complexity),
            "by_database": self._bucket_rates(by_database),
            "failure_examples": failures[:50],
        }

    def _predict(self, row: dict[str, Any], schema_mode: str) -> dict[str, Any]:
        if self.predictor is None:
            return row.get("predicted_query_ir") or row.get("query_ir") or {}
        return self.predictor(row, schema_mode=schema_mode)

    @staticmethod
    def _metrics(gold: dict[str, Any], pred: dict[str, Any], row: dict[str, Any]) -> dict[str, bool]:
        gold_joins = gold.get("joins") or []
        pred_joins = pred.get("joins") or []
        return {
            "intent_accuracy": gold.get("intent") == pred.get("intent"),
            "template_accuracy": gold.get("template_id") == pred.get("template_id"),
            "base_table_accuracy": gold.get("base_table") == pred.get("base_table"),
            "metric_accuracy": _projection(gold, "metrics", ["aggregation", "expression"]) == _projection(pred, "metrics", ["aggregation", "expression"]),
            "dimension_accuracy": _projection(gold, "dimensions", ["expression"]) == _projection(pred, "dimensions", ["expression"]),
            "filter_accuracy": _projection(gold, "filters", ["expression", "operator", "value"]) == _projection(pred, "filters", ["expression", "operator", "value"]),
            "date_filter_accuracy": _projection(gold, "date_filters", ["date_expression", "filter_type", "start_date", "end_date", "date_grain"]) == _projection(pred, "date_filters", ["date_expression", "filter_type", "start_date", "end_date", "date_grain"]),
            "join_accuracy": _projection(gold, "joins", ["condition"]) == _projection(pred, "joins", ["condition"]),
            "no_unnecessary_join": not pred_joins if not gold_joins else True,
            "query_ir_validity": bool(row.get("ir_validation", {}).get("is_valid", True)),
            "sql_validation": bool(row.get("sql_validation", {}).get("is_valid", row.get("sql_validation", {}).get("ok", True))),
            "structural_sql_match": normalize_sql(row.get("source_sql")) == normalize_sql(row.get("rendered_sql")) if row.get("source_sql") and row.get("rendered_sql") else True,
        }

    @staticmethod
    def _bucket_rates(buckets: dict[str, Counter]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for name, counter in buckets.items():
            total = counter.get("total", 0)
            result[name] = {
                f"{key}_rate": value / total
                for key, value in counter.items()
                if key != "total" and total
            }
            result[name]["total_examples"] = total
        return result


def _projection(ir: dict[str, Any], section: str, keys: list[str]) -> list[tuple[Any, ...]]:
    return sorted(tuple(item.get(key) for key in keys) for item in ir.get(section) or [])


def normalize_sql(value: str | None) -> str:
    return " ".join(str(value or "").lower().split())
