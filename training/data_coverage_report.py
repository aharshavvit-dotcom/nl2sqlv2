"""Training data coverage report.

Reports positive support per task head across the training corpus,
marking heads below minimum support as LOW_SUPPORT_DIAGNOSTIC_ONLY.

Gate: every active head must have adequate positive + negative support.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimum positive examples per head for meaningful training
MINIMUM_SUPPORT = 200

# All label keys that the neural model trains on
LABEL_KEYS = [
    "intent_label",
    "metric_aggregation_label",
    "metric_expression_type_label",
    "date_grain_label",
    "date_filter_type_label",
    "filter_operator_label",
    "order_direction_label",
    "limit_bucket_label",
    "base_table_index",
    "metric_column_index",
    "dimension_column_index",
    "date_column_index",
    "filter_column_index",
]

# Task mask keys
TASK_MASK_KEYS = [
    "capability", "safety", "table", "column", "aggregation",
    "filter", "join_edge", "complexity", "contrastive_schema_linking",
    "subquery", "window", "set_operation", "full_query_ir",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_coverage(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze label and mask coverage across training examples."""
    total = len(examples)

    # Intent distribution
    intent_counts: dict[str, int] = {}
    for row in examples:
        ir = row.get("query_ir") or {}
        intent = ir.get("intent", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # Dataset distribution
    dataset_counts: dict[str, int] = {}
    for row in examples:
        ds = row.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

    # Label support: count non-negative labels per key
    label_support: dict[str, int] = {}
    label_value_dist: dict[str, dict[str, int]] = {}
    for row in examples:
        ir = row.get("query_ir") or {}
        # Check which fields would produce non-negative labels
        if ir.get("intent"):
            label_support["intent_label"] = label_support.get("intent_label", 0) + 1
        if ir.get("base_table"):
            label_support["base_table_index"] = label_support.get("base_table_index", 0) + 1
        metrics = ir.get("metrics") or []
        if metrics:
            label_support["metric_column_index"] = label_support.get("metric_column_index", 0) + 1
            if any(m.get("aggregation") for m in metrics):
                label_support["metric_aggregation_label"] = label_support.get("metric_aggregation_label", 0) + 1
            if any(m.get("expression_type") or m.get("expression") for m in metrics):
                label_support["metric_expression_type_label"] = label_support.get("metric_expression_type_label", 0) + 1
        dimensions = ir.get("dimensions") or []
        if dimensions:
            label_support["dimension_column_index"] = label_support.get("dimension_column_index", 0) + 1
        filters = ir.get("filters") or []
        if filters:
            label_support["filter_column_index"] = label_support.get("filter_column_index", 0) + 1
            if any(f.get("operator") for f in filters):
                label_support["filter_operator_label"] = label_support.get("filter_operator_label", 0) + 1
        date_filters = ir.get("date_filters") or []
        if date_filters:
            label_support["date_column_index"] = label_support.get("date_column_index", 0) + 1
            if any(d.get("date_grain") for d in date_filters):
                label_support["date_grain_label"] = label_support.get("date_grain_label", 0) + 1
            if any(d.get("filter_type") for d in date_filters):
                label_support["date_filter_type_label"] = label_support.get("date_filter_type_label", 0) + 1
        order_by = ir.get("order_by") or []
        if order_by:
            label_support["order_direction_label"] = label_support.get("order_direction_label", 0) + 1
        limit = ir.get("limit")
        if limit is not None:
            label_support["limit_bucket_label"] = label_support.get("limit_bucket_label", 0) + 1

    # Task mask coverage
    mask_coverage: dict[str, int] = {}
    for row in examples:
        masks = row.get("task_masks") or {}
        if isinstance(masks, dict):
            for key in TASK_MASK_KEYS:
                if masks.get(key):
                    mask_coverage[key] = mask_coverage.get(key, 0) + 1
        elif isinstance(masks, list):
            for i, key in enumerate(TASK_MASK_KEYS):
                if i < len(masks) and masks[i]:
                    mask_coverage[key] = mask_coverage.get(key, 0) + 1

    # Head status
    head_status: dict[str, dict[str, Any]] = {}
    for label_key in LABEL_KEYS:
        support = label_support.get(label_key, 0)
        status = "ACTIVE" if support >= MINIMUM_SUPPORT else "LOW_SUPPORT_DIAGNOSTIC_ONLY"
        head_status[label_key] = {
            "positive_support": support,
            "support_rate": support / max(total, 1),
            "status": status,
            "minimum_required": MINIMUM_SUPPORT,
        }

    # Supervision level estimate
    full_supervision = sum(
        1 for row in examples
        if (row.get("query_ir") or {}).get("intent")
        and (row.get("query_ir") or {}).get("base_table")
    )

    return {
        "total_examples": total,
        "full_supervision_count": full_supervision,
        "intent_distribution": dict(sorted(intent_counts.items(), key=lambda x: -x[1])),
        "dataset_distribution": dict(sorted(dataset_counts.items(), key=lambda x: -x[1])),
        "label_positive_support": label_support,
        "task_mask_coverage": mask_coverage,
        "head_status": head_status,
        "heads_below_minimum_support": [
            k for k, v in head_status.items() if v["status"] == "LOW_SUPPORT_DIAGNOSTIC_ONLY"
        ],
        "heads_above_minimum_support": [
            k for k, v in head_status.items() if v["status"] == "ACTIVE"
        ],
        "gate_passed": all(
            v["status"] == "ACTIVE" for v in head_status.values()
        ),
    }


def main() -> None:
    data_files = {
        "generic_ir_train": ROOT / "data" / "processed" / "generic_ir_train.jsonl",
        "generic_ir_validation": ROOT / "data" / "processed" / "generic_ir_validation.jsonl",
        "training_data_ir": ROOT / "training_data" / "ir_training_examples.jsonl",
    }

    all_reports = {}
    for name, path in data_files.items():
        if not path.exists() or path.stat().st_size == 0:
            print(f"Skipping {name}: not found or empty")
            continue

        print(f"\n=== {name} ({path.name}) ===")
        examples = load_jsonl(path)
        report = analyze_coverage(examples)
        all_reports[name] = report

        print(f"  Total: {report['total_examples']}")
        print(f"  Full supervision: {report['full_supervision_count']}")
        print(f"  Intent distribution: {report['intent_distribution']}")
        print(f"  Dataset distribution: {report['dataset_distribution']}")
        print(f"\n  Head status:")
        for key, status in report["head_status"].items():
            marker = "OK" if status["status"] == "ACTIVE" else "LOW"
            print(f"    [{marker:3s}] {key}: {status['positive_support']} ({status['support_rate']:.1%})")
        print(f"\n  Gate: {'PASSED' if report['gate_passed'] else 'FAILED'}")
        if report["heads_below_minimum_support"]:
            print(f"  Low-support heads: {report['heads_below_minimum_support']}")

    output_path = ROOT / "artifacts" / "training_data" / "data_coverage_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\nCoverage report written to {output_path}")


if __name__ == "__main__":
    main()
