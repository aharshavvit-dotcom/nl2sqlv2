"""CLI: Run gold comparison between predictions and gold labels.

Usage
-----
python training/run_gold_comparison.py \\
  --predictions artifacts/self_training/iteration_0/predictions.jsonl \\
  --gold data/processed/generic_ir_validation.jsonl \\
  --output artifacts/self_training/comparison_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        description="Compare predictions against gold labels using the GoldComparator.",
    )
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "artifacts" / "self_training" / "iteration_0" / "predictions.jsonl",
                        help="Path to predictions JSONL file.")
    parser.add_argument("--gold", type=Path,
                        default=ROOT / "data" / "processed" / "generic_ir_validation.jsonl",
                        help="Path to gold labels JSONL file.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "self_training" / "comparison_report.json",
                        help="Path to write the comparison report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    predictions = _read_jsonl(args.predictions)
    gold = _read_jsonl(args.gold)

    if not predictions:
        print(f"WARNING: No predictions found at {args.predictions}")
    if not gold:
        print(f"WARNING: No gold labels found at {args.gold}")

    comparator = GoldComparator()
    report = comparator.compare_batch(predictions, gold)

    output = {
        "total": report.total,
        "exact_matches": report.exact_matches,
        "partial_matches": report.partial_matches,
        "failures": report.failures,
        "exact_match_rate": report.exact_matches / max(report.total, 1),
        "partial_match_rate": report.partial_matches / max(report.total, 1),
        "field_accuracy": report.field_accuracy,
        "per_example_summary": [
            {
                "example_id": r.example_id,
                "match_score": r.match_score,
                "is_exact_match": r.is_exact_match,
                "is_partial_match": r.is_partial_match,
                "mismatched_fields": [f for f, m in r.field_matches.items() if not m],
            }
            for r in report.per_example[:100]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
