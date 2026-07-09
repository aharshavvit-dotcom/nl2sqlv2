"""Stage 0: Semantic metric audit script.

Reads the existing evaluation report and produces a detailed audit of each
semantic metric, verifying denominators, checking for instrumentation bugs,
and producing the semantic_metric_audit.jsonl artifact.

Usage:
    python -m evaluation.metric_audit [--report PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def audit_semantic_metrics(report_path: Path, output_path: Path) -> dict[str, Any]:
    """Audit semantic metric computations from an evaluation report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))

    test_perf = report.get("test_performance", {})
    per_example = test_perf.get("per_example", [])
    summary = test_perf.get("summary", {})

    total = len(per_example)
    audit_records: list[dict[str, Any]] = []
    counters: dict[str, dict[str, int]] = {
        "filter_column_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
        "filter_value_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
        "filter_value_extraction_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
        "filter_column_top1_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
        "filter_column_top3_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
        "dimension_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
        "projection_exact_match": {"applicable": 0, "true": 0, "false": 0, "none": 0},
    }

    issues_found: list[str] = []

    for i, item in enumerate(per_example):
        filter_linking = item.get("filter_linking") or {}
        dimension_linking = item.get("dimension_linking") or {}
        projection = item.get("projection") or {}
        semantic_pass = item.get("semantic_pass") or {}

        record: dict[str, Any] = {
            "example_index": i,
            "example_id": item.get("example_id"),
            "question": item.get("question"),
        }

        # --- Filter diagnostics ---
        gold_filter_col = filter_linking.get("gold_filter_column")
        pred_filter_col = filter_linking.get("predicted_filter_column")
        gold_filter_val = filter_linking.get("gold_filter_value")
        pred_filter_val = filter_linking.get("predicted_filter_value")
        filter_col_match = filter_linking.get("filter_column_match")
        filter_val_match = filter_linking.get("filter_value_match")
        fv_extract_match = filter_linking.get("filter_value_extraction_match")
        top1_match = filter_linking.get("filter_column_top1_match")
        top3_match = filter_linking.get("filter_column_top3_match")

        record.update({
            "gold_filter_column": gold_filter_col,
            "gold_filter_value": gold_filter_val,
            "predicted_filter_column": pred_filter_col,
            "predicted_filter_value": pred_filter_val,
            "filter_column_match": filter_col_match,
            "filter_value_match": filter_val_match,
            "filter_value_extraction_match": fv_extract_match,
            "filter_column_top1_match": top1_match,
            "filter_column_top3_match": top3_match,
            "filter_value_candidates": filter_linking.get("filter_value_candidates") or [],
            "gold_column_present_in_candidates": _gold_in_candidates(
                gold_filter_col,
                filter_linking.get("filter_value_candidates") or [],
            ),
            "linking_method": filter_linking.get("linking_method"),
            "linking_confidence": filter_linking.get("linking_confidence"),
        })

        # Count each metric
        for key in ["filter_column_match", "filter_value_match", "filter_value_extraction_match",
                     "filter_column_top1_match", "filter_column_top3_match"]:
            val = filter_linking.get(key)
            _count(counters[key], val)

        # --- Dimension diagnostics ---
        gold_dim = dimension_linking.get("gold_dimension")
        pred_dim = dimension_linking.get("predicted_dimension")
        dim_match = dimension_linking.get("dimension_match")
        record.update({
            "gold_dimension": gold_dim,
            "predicted_dimension": pred_dim,
            "dimension_match": dim_match,
        })
        _count(counters["dimension_match"], dim_match)

        # --- Projection diagnostics ---
        gold_proj = projection.get("gold_columns") or []
        pred_proj = projection.get("predicted_columns") or []
        proj_exact = projection.get("exact_match")
        record.update({
            "gold_projection_columns": gold_proj,
            "predicted_projection_columns": pred_proj,
            "projection_exact_match": proj_exact,
            "has_extra_columns": projection.get("has_extra_columns"),
            "default_projection_used": projection.get("default_projection_used"),
        })
        _count(counters["projection_exact_match"], proj_exact if gold_proj else None)

        # --- Semantic pass ---
        record["semantic_pass"] = semantic_pass
        audit_records.append(record)

    # --- Anomaly detection ---
    for key, counts in counters.items():
        reported_rate = summary.get(f"{key}_rate", summary.get(f"{key}_accuracy_rate"))
        computed_rate = counts["true"] / counts["applicable"] if counts["applicable"] else 0.0
        if reported_rate is not None and abs(float(reported_rate) - computed_rate) > 0.001:
            issues_found.append(
                f"DENOMINATOR_MISMATCH: {key}: reported_rate={reported_rate:.4f}, "
                f"computed_rate={computed_rate:.4f}, true={counts['true']}, "
                f"applicable={counts['applicable']}, none={counts['none']}"
            )

    # Specific zero-value checks
    if counters["filter_column_top1_match"]["applicable"] == 0:
        issues_found.append(
            "ZERO_DENOMINATOR: filter_column_top1_match has 0 applicable cases. "
            "This likely means filter_value_candidates are empty for all rows."
        )
    if counters["filter_value_extraction_match"]["applicable"] == 0:
        issues_found.append(
            "ZERO_DENOMINATOR: filter_value_extraction_match has 0 applicable cases. "
            "This likely means no filter values were extracted by the slot resolver."
        )

    # Qualified vs unqualified column name check
    qualified_gold = sum(1 for r in audit_records if r["gold_filter_column"] and "." in str(r["gold_filter_column"]))
    unqualified_gold = sum(1 for r in audit_records if r["gold_filter_column"] and "." not in str(r["gold_filter_column"]))
    qualified_pred = sum(1 for r in audit_records if r["predicted_filter_column"] and "." in str(r["predicted_filter_column"]))
    unqualified_pred = sum(1 for r in audit_records if r["predicted_filter_column"] and "." not in str(r["predicted_filter_column"]))

    if qualified_gold > 0 and unqualified_pred > 0:
        issues_found.append(
            f"NAME_FORMAT_MISMATCH: gold uses qualified column names ({qualified_gold} cases) "
            f"but predictions use unqualified names ({unqualified_pred} cases). "
            "This will cause false negatives in filter_column_match."
        )

    # --- Write outputs ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in audit_records:
            f.write(json.dumps(record, default=str) + "\n")

    audit_summary = {
        "total_examples": total,
        "metric_counters": counters,
        "issues_found": issues_found,
        "issue_count": len(issues_found),
        "name_format": {
            "qualified_gold_filter_columns": qualified_gold,
            "unqualified_gold_filter_columns": unqualified_gold,
            "qualified_predicted_filter_columns": qualified_pred,
            "unqualified_predicted_filter_columns": unqualified_pred,
        },
        "reported_rates": {
            "filter_column_accuracy_rate": summary.get("filter_column_accuracy_rate"),
            "filter_value_accuracy_rate": summary.get("filter_value_accuracy_rate"),
            "filter_value_extraction_accuracy_rate": summary.get("filter_value_extraction_accuracy_rate"),
            "filter_column_top1_accuracy_rate": summary.get("filter_column_top1_accuracy_rate"),
            "filter_column_top3_accuracy_rate": summary.get("filter_column_top3_accuracy_rate"),
            "dimension_column_accuracy_rate": summary.get("dimension_column_accuracy_rate"),
            "projection_exact_match_rate": summary.get("projection_exact_match_rate"),
        },
    }

    summary_path = output_path.parent / "semantic_metric_audit_summary.json"
    summary_path.write_text(json.dumps(audit_summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== Semantic Metric Audit ===")
    print(f"Total examples: {total}")
    print(f"Issues found: {len(issues_found)}")
    for issue in issues_found:
        print(f"  ⚠ {issue}")
    print(f"\nMetric counters:")
    for key, counts in counters.items():
        rate = counts["true"] / counts["applicable"] if counts["applicable"] else 0.0
        print(f"  {key}: {rate:.4f} ({counts['true']}/{counts['applicable']}, {counts['none']} excluded)")
    print(f"\nAudit written to: {output_path}")
    print(f"Summary written to: {summary_path}")

    return audit_summary


def _count(counter: dict[str, int], value: bool | None) -> None:
    if value is None:
        counter["none"] += 1
    elif value:
        counter["applicable"] += 1
        counter["true"] += 1
    else:
        counter["applicable"] += 1
        counter["false"] += 1


def _gold_in_candidates(
    gold_column: str | None,
    candidates: list[dict[str, Any]],
) -> bool | None:
    if gold_column is None:
        return None
    gold_norm = str(gold_column).strip().lower()
    for candidate in candidates:
        for col in candidate.get("candidate_columns") or []:
            col_name = str(col.get("column") or "").strip().lower()
            if col_name == gold_norm or col_name.endswith(f".{gold_norm}") or gold_norm.endswith(f".{col_name}"):
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit semantic metric computations")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json",
        help="Path to the evaluation report JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "evaluation" / "semantic_metric_audit.jsonl",
        help="Output path for the audit JSONL",
    )
    args = parser.parse_args()
    audit_semantic_metrics(args.report, args.output)
