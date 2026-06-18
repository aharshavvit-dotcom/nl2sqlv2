from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


class CorpusQualityAnalyzer:
    def analyze(self, examples: list[dict[str, Any]], unsupported: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(examples) + len(unsupported)
        join_counts = [len((row.get("query_ir") or {}).get("joins") or []) for row in examples]
        filter_counts = [len((row.get("query_ir") or {}).get("filters") or []) for row in examples]
        date_filter_counts = [len((row.get("query_ir") or {}).get("date_filters") or []) for row in examples]
        return {
            "total_examples": total,
            "supported_examples": len(examples),
            "unsupported_examples": len(unsupported),
            "conversion_success_rate": len(examples) / total if total else 0.0,
            "query_ir_validation_rate": self._rate(examples, "ir_validation"),
            "sql_validation_rate": self._rate(examples, "sql_validation"),
            "roundtrip_validation_rate": self._rate(examples, "roundtrip_validation"),
            "dataset_distribution": dict(Counter(row.get("dataset_name") for row in examples)),
            "intent_distribution": dict(Counter(row.get("intent") for row in examples)),
            "complexity_distribution": dict(Counter(row.get("complexity") for row in examples)),
            "database_distribution": dict(Counter(row.get("db_id") for row in examples)),
            "schema_size_distribution": self._schema_sizes(examples),
            "join_count_distribution": dict(Counter(join_counts)),
            "filter_count_distribution": dict(Counter(filter_counts)),
            "date_filter_count_distribution": dict(Counter(date_filter_counts)),
            "avg_join_count": mean(join_counts) if join_counts else 0.0,
            "metric_expression_distribution": dict(Counter(self._metric_expressions(examples))),
            "unsupported_reason_distribution": dict(Counter(row.get("unsupported_reason") for row in unsupported)),
        }

    @staticmethod
    def _rate(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for row in rows if (row.get(key) or {}).get("is_valid", True)) / len(rows)

    @staticmethod
    def _schema_sizes(rows: list[dict[str, Any]]) -> dict[str, int]:
        buckets = Counter()
        for row in rows:
            tables = ((row.get("schema") or {}).get("tables") or {})
            size = len(tables) if isinstance(tables, dict) else 0
            bucket = "small" if size <= 5 else ("medium" if size <= 15 else "large")
            buckets[bucket] += 1
        return dict(buckets)

    @staticmethod
    def _metric_expressions(rows: list[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        for row in rows:
            for metric in (row.get("query_ir") or {}).get("metrics") or []:
                values.append(str(metric.get("expression") or metric.get("name") or "none"))
        return values
