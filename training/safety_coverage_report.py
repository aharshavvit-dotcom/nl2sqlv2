"""Safety supervision diagnosis for NL2SQL training corpus.

Scans the training dataset to determine why the safety head receives
zero effective supervision. Reports:
- safety_positive_count / safety_negative_count
- safety_mask_active_count / safety_mask_zero_count
- safety_label distribution
- task_mask["safety"] coverage

This is a diagnostic-only script. It does NOT modify any data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_safety_coverage(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze safety label and mask coverage across examples."""
    total = len(examples)

    # Task mask analysis
    safety_mask_active = 0
    safety_mask_zero = 0
    safety_mask_missing = 0

    # Label analysis
    safety_label_values: dict[str, int] = {}
    has_safety_label = 0
    missing_safety_label = 0

    # Query IR safety signal analysis
    has_safe_field = 0
    safe_true = 0
    safe_false = 0
    safe_missing = 0

    # Capability analysis
    has_safety_capability = 0

    for row in examples:
        # Check task_masks
        task_masks = row.get("task_masks") or {}
        if isinstance(task_masks, dict):
            safety_val = task_masks.get("safety")
            if safety_val is None:
                safety_mask_missing += 1
            elif safety_val:
                safety_mask_active += 1
            else:
                safety_mask_zero += 1
        elif isinstance(task_masks, list):
            # task_masks might be a list indexed by TASK_MASK_KEYS
            # "safety" is at index 1
            if len(task_masks) > 1 and task_masks[1]:
                safety_mask_active += 1
            else:
                safety_mask_zero += 1
        else:
            safety_mask_missing += 1

        # Check for explicit safety labels
        safety_label = row.get("safety_label") or row.get("is_safe")
        if safety_label is not None:
            has_safety_label += 1
            key = str(safety_label)
            safety_label_values[key] = safety_label_values.get(key, 0) + 1
        else:
            missing_safety_label += 1

        # Check query_ir for safety signals
        query_ir = row.get("query_ir") or {}
        if isinstance(query_ir, dict):
            if "is_safe" in query_ir:
                has_safe_field += 1
                if query_ir["is_safe"]:
                    safe_true += 1
                else:
                    safe_false += 1
            else:
                safe_missing += 1

        # Check capabilities
        capabilities = row.get("capabilities") or row.get("capability_labels") or []
        if isinstance(capabilities, dict):
            if capabilities.get("safety"):
                has_safety_capability += 1
        elif isinstance(capabilities, list):
            # Check if any capability mentions safety
            if any("safe" in str(c).lower() for c in capabilities):
                has_safety_capability += 1

    return {
        "total_examples": total,
        "task_mask_analysis": {
            "safety_mask_active": safety_mask_active,
            "safety_mask_zero": safety_mask_zero,
            "safety_mask_missing": safety_mask_missing,
        },
        "safety_label_analysis": {
            "has_safety_label": has_safety_label,
            "missing_safety_label": missing_safety_label,
            "label_distribution": safety_label_values,
        },
        "query_ir_safety_analysis": {
            "has_is_safe_field": has_safe_field,
            "is_safe_true": safe_true,
            "is_safe_false": safe_false,
            "is_safe_missing": safe_missing,
        },
        "capability_analysis": {
            "has_safety_capability": has_safety_capability,
        },
        "root_cause_diagnosis": _diagnose_root_cause(
            total, safety_mask_active, safety_mask_zero, safety_mask_missing,
            has_safety_label, safe_true, safe_false, safe_missing,
        ),
    }


def _diagnose_root_cause(
    total: int,
    mask_active: int,
    mask_zero: int,
    mask_missing: int,
    has_label: int,
    safe_true: int,
    safe_false: int,
    safe_missing: int,
) -> list[str]:
    """Generate root cause diagnosis for zero safety supervision."""
    causes = []

    if mask_active == 0:
        causes.append(
            "CRITICAL: safety_mask is zero/missing for ALL examples. "
            "The safety loss head is masked out for every training example, "
            "resulting in zero gradient regardless of label availability."
        )

    if has_label == 0:
        causes.append(
            "CRITICAL: No examples have explicit safety_label or is_safe field. "
            "Even if mask were active, there is no supervision signal."
        )

    if safe_true > 0 and safe_false == 0:
        causes.append(
            "WARNING: All examples with is_safe field are positive (safe). "
            "No negative (unsafe) examples exist. Binary classification "
            "requires both classes."
        )

    if mask_active > 0 and has_label == 0:
        causes.append(
            "ISSUE: Safety mask is active for some examples but no labels exist. "
            "The mask allows gradient flow but there is nothing to learn from."
        )

    if not causes:
        causes.append(
            f"Safety supervision appears available: mask_active={mask_active}, "
            f"labels={has_label}, safe_true={safe_true}, safe_false={safe_false}. "
            "Investigate loss computation code."
        )

    return causes


def main() -> None:
    # Scan all available training data files
    data_files = [
        ROOT / "data" / "processed" / "generic_ir_train.jsonl",
        ROOT / "data" / "processed" / "ir_training_examples.jsonl",
        ROOT / "training_data" / "ir_training_examples.jsonl",
    ]

    output_dir = ROOT / "artifacts" / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    for data_file in data_files:
        if not data_file.exists():
            print(f"Skipping {data_file.name}: not found")
            continue
        if data_file.stat().st_size == 0:
            print(f"Skipping {data_file.name}: empty")
            continue

        print(f"Analyzing {data_file.name}...")
        examples = load_jsonl(data_file)
        report = analyze_safety_coverage(examples)
        all_reports[data_file.name] = report

        print(f"  Total examples: {report['total_examples']}")
        print(f"  Safety mask active: {report['task_mask_analysis']['safety_mask_active']}")
        print(f"  Safety mask zero: {report['task_mask_analysis']['safety_mask_zero']}")
        print(f"  Safety mask missing: {report['task_mask_analysis']['safety_mask_missing']}")
        print(f"  Has safety label: {report['safety_label_analysis']['has_safety_label']}")
        print(f"  Query IR is_safe field: {report['query_ir_safety_analysis']['has_is_safe_field']}")
        for cause in report["root_cause_diagnosis"]:
            print(f"  -> {cause}")

    # Write combined report
    output_path = output_dir / "safety_coverage_report.json"
    output_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\nSafety coverage report written to {output_path}")


if __name__ == "__main__":
    main()
