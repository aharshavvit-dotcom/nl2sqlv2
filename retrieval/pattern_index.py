from __future__ import annotations

import re
from collections import Counter
from typing import Any


PATTERNS = {
    "show_records",
    "count_records",
    "count_by_dimension",
    "simple_filter",
    "metric_summary",
    "metric_by_dimension",
    "top_n_metric_by_dimension",
    "bottom_n_metric_by_dimension",
    "trend_by_date",
    "joined_records",
}


class PatternIndex:
    def __init__(self):
        self.pattern_counts: Counter[str] = Counter()
        self.examples_by_pattern: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in PATTERNS}

    def build(self, examples: list[dict[str, Any]]) -> None:
        self.pattern_counts.clear()
        self.examples_by_pattern = {pattern: [] for pattern in PATTERNS}
        for row in examples:
            pattern = row.get("intent") or row.get("template_id") or (row.get("query_ir") or {}).get("intent") or "unknown"
            if (row.get("query_ir") or {}).get("joins") and pattern == "show_records":
                pattern = "joined_records"
            if pattern not in PATTERNS:
                pattern = "metric_summary"
            self.pattern_counts[pattern] += 1
            self.examples_by_pattern.setdefault(pattern, []).append(row)

    def search_patterns(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        inferred = infer_pattern(question)
        values = []
        for pattern in PATTERNS:
            score = 1.0 if pattern == inferred else 0.15
            if pattern == "show_records" and inferred in {"simple_filter", "count_records"}:
                score = 0.35
            values.append({"pattern": pattern, "intent": pattern, "score": score, "count": self.pattern_counts.get(pattern, 0)})
        return sorted(values, key=lambda item: item["score"], reverse=True)[:top_k]


def infer_pattern(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ["top", "highest", "most"]):
        return "top_n_metric_by_dimension"
    if any(word in q for word in ["bottom", "lowest", "least"]):
        return "bottom_n_metric_by_dimension"
    if any(word in q for word in ["trend", "monthly", "by month", "by year"]):
        return "trend_by_date"
    if re.search(r"\b(count|how many|number of|total)\b", q):
        return "count_records" if " by " not in q else "count_by_dimension"
    if re.search(r"\bwhere\b|\b(status|role|category)\s+(is|=|equals)\b", q):
        return "simple_filter"
    if re.search(r"\bwith\b|\balong with\b|\band their\b", q):
        return "joined_records"
    if re.search(r"\bby\s+\w+", q):
        return "metric_by_dimension"
    if re.search(r"\b(list|show|display|view|fetch|get)\b", q):
        return "show_records"
    return "metric_summary"
