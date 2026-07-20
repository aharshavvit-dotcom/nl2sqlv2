"""Data accounting for NL2SQL training corpus.

Reconciles: RAW = TRAIN + VAL + MODEL_SELECTION + CALIBRATION + TEST + REJECTED
Reports exactly why every unused example is excluded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def main() -> None:
    processed = ROOT / "data" / "processed"
    training_data = ROOT / "training_data"

    # Count all available files
    accounting: dict[str, Any] = {
        "generated_at": None,
        "source_datasets": {},
        "processed_files": {},
        "split_reconciliation": {},
        "rejection_analysis": {},
        "data_utilization": {},
        "recommendations": [],
    }

    # Source dataset stats
    dataset_stats_path = processed / "dataset_stats.json"
    if dataset_stats_path.exists():
        ds_stats = json.loads(dataset_stats_path.read_text(encoding="utf-8"))
        accounting["source_datasets"] = {
            "raw_total": ds_stats.get("total_examples", 0),
            "supported": ds_stats.get("supported_examples", 0),
            "unsupported": ds_stats.get("unsupported_examples", 0),
            "by_dataset": ds_stats.get("by_dataset", {}),
            "by_template": ds_stats.get("by_template", {}),
            "by_split": ds_stats.get("by_split", {}),
            "unsupported_reasons": ds_stats.get("unsupported_reasons", {}),
        }

    # IR conversion stats
    ir_stats_path = processed / "ir_dataset_stats.json"
    if ir_stats_path.exists():
        ir_stats = json.loads(ir_stats_path.read_text(encoding="utf-8"))
        accounting["ir_conversion"] = {
            "total_attempted": ir_stats.get("total_examples", 0),
            "successful": ir_stats.get("successful_examples", 0),
            "unsupported": ir_stats.get("unsupported_examples", 0),
            "conversion_rate": ir_stats.get("conversion_success_rate", 0),
            "by_dataset": ir_stats.get("by_dataset", {}),
            "by_intent": ir_stats.get("by_intent", {}),
            "by_split": ir_stats.get("by_split", {}),
            "unsupported_reasons": ir_stats.get("by_unsupported_reason", {}),
        }

    # Count all processed files
    file_counts: dict[str, dict[str, Any]] = {}
    important_files = {
        "unified_examples.jsonl": "All examples before filtering",
        "supported_examples.jsonl": "Examples that pass SQL->IR conversion",
        "unsupported_examples.jsonl": "Examples that fail conversion",
        "generic_ir_train.jsonl": "Active neural training split",
        "generic_ir_validation.jsonl": "Active neural validation split",
        "generic_ir_test.jsonl": "Active neural test split",
        "generic_ir_model_selection_validation.jsonl": "Model selection validation",
        "generic_ir_unseen_db_test.jsonl": "Unseen-DB generalization test",
        "generic_ir_hard_negatives.jsonl": "Hard negative examples",
        "generic_ir_partial_supervision.jsonl": "Partially supervised examples",
        "generic_ir_capability_annotations.jsonl": "Capability-annotated examples",
        "generic_ir_unsupported.jsonl": "Unsupported examples (rejected)",
        "generic_ir_controlled_execution_test.jsonl": "Controlled execution test",
        "schema_registry.jsonl": "Schema registry",
    }

    for filename, description in important_files.items():
        path = processed / filename
        if path.exists():
            lines = count_lines(path)
            file_counts[filename] = {
                "count": lines,
                "size_bytes": path.stat().st_size,
                "description": description,
            }
        else:
            file_counts[filename] = {
                "count": 0,
                "size_bytes": 0,
                "description": description,
                "status": "MISSING",
            }
    accounting["processed_files"] = file_counts

    # Split reconciliation
    train_count = file_counts.get("generic_ir_train.jsonl", {}).get("count", 0)
    val_count = file_counts.get("generic_ir_validation.jsonl", {}).get("count", 0)
    test_count = file_counts.get("generic_ir_test.jsonl", {}).get("count", 0)
    unseen_count = file_counts.get("generic_ir_unseen_db_test.jsonl", {}).get("count", 0)
    model_sel_count = file_counts.get("generic_ir_model_selection_validation.jsonl", {}).get("count", 0)
    partial_count = file_counts.get("generic_ir_partial_supervision.jsonl", {}).get("count", 0)
    unsupported_count = file_counts.get("generic_ir_unsupported.jsonl", {}).get("count", 0)

    total_accounted = train_count + val_count + test_count + unseen_count + model_sel_count
    raw_total = accounting.get("source_datasets", {}).get("raw_total", 0)

    accounting["split_reconciliation"] = {
        "train": train_count,
        "validation": val_count,
        "test": test_count,
        "unseen_db_test": unseen_count,
        "model_selection_validation": model_sel_count,
        "partial_supervision": partial_count,
        "unsupported_rejected": unsupported_count,
        "total_in_splits": total_accounted,
        "raw_source_total": raw_total,
        "accounted_ratio": total_accounted / max(raw_total, 1),
        "unaccounted": raw_total - total_accounted - unsupported_count - partial_count,
    }

    # Data utilization analysis
    accounting["data_utilization"] = {
        "raw_available": raw_total,
        "actively_training": train_count,
        "utilization_rate": train_count / max(raw_total, 1),
        "with_partial_supervision": train_count + partial_count,
        "partial_utilization_rate": (train_count + partial_count) / max(raw_total, 1),
        "wasted_valid_examples": partial_count,
        "wasted_due_to_unsupported": unsupported_count,
    }

    # Recommendations
    recs = []
    if train_count < 3000:
        recs.append(
            f"CRITICAL: Only {train_count} examples in active training. "
            f"Target: maximize valid supervision from {raw_total} available."
        )
    if partial_count > 0:
        recs.append(
            f"OPPORTUNITY: {partial_count} partially supervised examples available "
            f"but not used in training. Enable partial supervision with task masks."
        )
    if unsupported_count > 0:
        reasons = accounting.get("ir_conversion", {}).get("unsupported_reasons", {})
        if reasons:
            top_reason = max(reasons, key=reasons.get)
            recs.append(
                f"RECOVERY: {unsupported_count} unsupported examples. "
                f"Top reason: {top_reason} ({reasons[top_reason]}). "
                f"Consider expanding SQL->IR converter to support this pattern."
            )
    accounting["recommendations"] = recs

    # Add timestamp
    from datetime import datetime, timezone
    accounting["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Write report
    output_path = ROOT / "artifacts" / "training_data" / "data_accounting.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(accounting, indent=2), encoding="utf-8")

    # Print summary
    print("=== Data Accounting Summary ===")
    print(f"Raw available: {raw_total}")
    print(f"Actively training: {train_count}")
    print(f"Utilization rate: {train_count / max(raw_total, 1):.1%}")
    print(f"Partial supervision available: {partial_count}")
    print(f"Unsupported/rejected: {unsupported_count}")
    print(f"\nSplit counts:")
    for split, count in accounting["split_reconciliation"].items():
        if isinstance(count, int):
            print(f"  {split}: {count}")
    print(f"\nRecommendations:")
    for rec in recs:
        print(f"  - {rec}")
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
