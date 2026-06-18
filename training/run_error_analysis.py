"""CLI: Run error analysis on a comparison report.

Usage
-----
python training/run_error_analysis.py \\
  --predictions artifacts/self_training/iteration_0/predictions.jsonl \\
  --gold data/processed/generic_ir_validation.jsonl \\
  --output artifacts/self_training/error_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_training.error_classifier import ErrorClassifier
from self_training.gold_comparator import GoldComparator


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify prediction errors into actionable categories.",
    )
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "artifacts" / "self_training" / "iteration_0" / "predictions.jsonl",
                        help="Path to predictions JSONL file.")
    parser.add_argument("--gold", type=Path,
                        default=ROOT / "data" / "processed" / "generic_ir_validation.jsonl",
                        help="Path to gold labels JSONL file.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "self_training" / "error_report.json",
                        help="Path to write the error report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    predictions = _read_jsonl(args.predictions)
    gold = _read_jsonl(args.gold)

    if not predictions:
        print(f"WARNING: No predictions found at {args.predictions}")

    # Step 1: Compare
    comparator = GoldComparator()
    comparison = comparator.compare_batch(predictions, gold)

    # Step 2: Classify errors
    classifier = ErrorClassifier()
    error_report = classifier.classify_batch(comparison.per_example, predictions)

    output = {
        "total_errors": error_report.total_errors,
        "by_category": error_report.by_category,
        "by_severity": error_report.by_severity,
        "by_dataset": error_report.by_dataset,
        "top_error_categories": error_report.top_error_categories,
        "sample_errors": [
            {
                "example_id": ec.example_id,
                "categories": [c.value for c in ec.categories],
                "severity": ec.severity,
                "suggested_fix_type": ec.suggested_fix_type,
            }
            for ec in error_report.classifications[:50]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
