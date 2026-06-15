from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .schema_linearizer import schema_from_example


class IRDatasetQualityAnalyzer:
    def analyze(self, input_path: str) -> dict:
        rows = _load_jsonl(Path(input_path))
        intent_distribution: Counter[str] = Counter()
        dataset_distribution: Counter[str] = Counter()
        schema_sizes = []
        missing_label_counts: Counter[str] = Counter()
        unsupported_reasons: Counter[str] = Counter()
        metric_expression_distribution: Counter[str] = Counter()
        filter_distribution: Counter[str] = Counter()
        date_filter_distribution: Counter[str] = Counter()
        join_count_distribution: Counter[str] = Counter()
        product_revenue_examples = 0

        for row in rows:
            query_ir = row.get("query_ir") or {}
            intent = query_ir.get("template_id") or query_ir.get("intent") or row.get("intent") or "unknown"
            intent_distribution[str(intent)] += 1
            dataset_distribution[str(row.get("dataset_name") or row.get("dataset") or "unknown")] += 1
            schema = schema_from_example(row)
            table_count = len((schema.get("tables") or {}))
            column_count = sum(len((info.get("columns") if isinstance(info, dict) else {}) or {}) for info in (schema.get("tables") or {}).values())
            schema_sizes.append({"tables": table_count, "columns": column_count})
            if not query_ir:
                missing_label_counts["query_ir"] += 1
                unsupported_reasons[str(row.get("unsupported_reason") or "missing_query_ir")] += 1
                continue
            if not query_ir.get("base_table"):
                missing_label_counts["base_table"] += 1
            if intent in {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "trend_by_date"} and not query_ir.get("metrics"):
                missing_label_counts["metrics"] += 1
            metric_expression_distribution[_metric_expression_type(query_ir)] += 1
            filter_distribution[str(len(query_ir.get("filters") or []))] += 1
            date_filter_distribution[str(len(query_ir.get("date_filters") or []))] += 1
            join_count_distribution[str(len(query_ir.get("joins") or []))] += 1
            if _is_product_revenue(query_ir, row.get("question", "")):
                product_revenue_examples += 1

        return {
            "total_rows": len(rows),
            "intent_distribution": dict(intent_distribution),
            "dataset_distribution": dict(dataset_distribution),
            "schema_size_distribution": _schema_size_summary(schema_sizes),
            "missing_label_counts": dict(missing_label_counts),
            "unsupported_reason_distribution": dict(unsupported_reasons),
            "metric_expression_distribution": dict(metric_expression_distribution),
            "filter_distribution": dict(filter_distribution),
            "date_filter_distribution": dict(date_filter_distribution),
            "join_count_distribution": dict(join_count_distribution),
            "product_revenue_examples_count": product_revenue_examples,
        }


def _metric_expression_type(query_ir: dict[str, Any]) -> str:
    metrics = query_ir.get("metrics") or []
    if not metrics:
        return "none"
    expression = str(metrics[0].get("expression") or "")
    if expression == "*":
        return "count_star"
    lowered = expression.lower()
    if "quantity" in lowered and "price" in lowered:
        return "product_revenue_expression"
    return "column_expression" if "." in expression else "other"


def _is_product_revenue(query_ir: dict[str, Any], question: str) -> bool:
    text = question.lower()
    dimension_text = " ".join(str(item.get("name") or item.get("column") or "") for item in query_ir.get("dimensions") or []).lower()
    metric_text = " ".join(str(item.get("name") or item.get("expression") or "") for item in query_ir.get("metrics") or []).lower()
    return ("product" in text or "product" in dimension_text) and any(token in text or token in metric_text for token in ["sales", "revenue"])


def _schema_size_summary(values: list[dict[str, int]]) -> dict[str, float | int]:
    if not values:
        return {"rows": 0, "avg_tables": 0.0, "avg_columns": 0.0, "max_tables": 0, "max_columns": 0}
    return {
        "rows": len(values),
        "avg_tables": sum(item["tables"] for item in values) / len(values),
        "avg_columns": sum(item["columns"] for item in values) / len(values),
        "max_tables": max(item["tables"] for item in values),
        "max_columns": max(item["columns"] for item in values),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
