from __future__ import annotations

from collections import defaultdict
from typing import Any


PHASES = [
    ("phase_1_wikisql_simple", {"datasets": {"wikisql"}, "complexities": {"simple"}}),
    ("phase_2_spider_simple_medium", {"datasets": {"spider"}, "complexities": {"simple", "medium"}}),
    ("phase_3_bird_mini_supported", {"datasets": {"bird-mini", "bird", "bird-mini-dev"}, "complexities": {"simple", "medium", "hard"}}),
    ("phase_4_mixed_finetune", {"datasets": None, "complexities": {"simple", "medium", "hard"}}),
]


class CurriculumPlanner:
    def split(self, rows: list[dict[str, Any]], max_examples_per_phase: int | None = 200) -> list[dict[str, Any]]:
        enriched = [{**row, "_dataset": _dataset_name(row), "_complexity": classify_complexity(row)} for row in rows]
        phases = []
        for name, rule in PHASES:
            datasets = rule["datasets"]
            complexities = rule["complexities"]
            selected = [
                row
                for row in enriched
                if (datasets is None or row["_dataset"] in datasets)
                and row["_complexity"] in complexities
            ]
            if not selected and name != "phase_4_mixed_finetune":
                continue
            if not selected:
                selected = list(enriched)
            if max_examples_per_phase is not None:
                selected = selected[:max_examples_per_phase]
            phases.append(
                {
                    "name": name,
                    "rows": [_strip_private(row) for row in selected],
                    "example_count": len(selected),
                    "by_complexity": _counts(row["_complexity"] for row in selected),
                    "by_dataset": _counts(row["_dataset"] for row in selected),
                }
            )
        return phases


def classify_complexity(row: dict[str, Any]) -> str:
    query_ir = row.get("query_ir") or {}
    joins = query_ir.get("joins") or []
    filters = query_ir.get("filters") or []
    date_filters = query_ir.get("date_filters") or []
    group_by = query_ir.get("group_by") or []
    order_by = query_ir.get("order_by") or []
    metrics = query_ir.get("metrics") or []
    expression = " ".join(str(metric.get("expression") or "") for metric in metrics if isinstance(metric, dict)).lower()
    join_count = len(joins)
    if not joins:
        required = query_ir.get("required_tables") or []
        join_count = max(0, len(set(required)) - 1)
    if join_count == 0 and not group_by and not filters and not date_filters:
        return "simple"
    if join_count <= 1 and (group_by or order_by) and len(filters) <= 1 and len(date_filters) <= 1:
        return "medium"
    if "order_items.quantity" in expression and "order_items.price" in expression:
        return "hard"
    return "hard" if join_count > 1 or filters or date_filters else "medium"


def _dataset_name(row: dict[str, Any]) -> str:
    raw = str(row.get("dataset_name") or row.get("dataset") or row.get("example_id") or "unknown").lower()
    if "wikisql" in raw:
        return "wikisql"
    if "spider" in raw:
        return "spider"
    if "bird" in raw:
        return "bird-mini"
    return raw.split(":", 1)[0]


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}
