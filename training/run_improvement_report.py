"""CLI: Generate an improvement report from self-training history.

Usage
-----
python training/run_improvement_report.py \\
  --history artifacts/self_training/improvement_history.json \\
  --output artifacts/self_training/improvement_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_training.improvement_tracker import ImprovementTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an improvement report from self-training iteration history.",
    )
    parser.add_argument("--history-dir", type=Path,
                        default=ROOT / "artifacts" / "self_training",
                        help="Directory containing improvement_history.json.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "self_training" / "improvement_report.json",
                        help="Path to write the improvement report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tracker = ImprovementTracker(args.history_dir)

    if tracker.iteration_count == 0:
        print("No iteration history found. Run the self-training loop first.")
        return 1

    report = tracker.generate_report()
    report_dict = report.to_dict()

    # Add human-readable summary
    report_dict["summary"] = {
        "total_iterations": tracker.iteration_count,
        "best_iteration": report.best_iteration,
        "converged": report.converged,
        "convergence_reason": report.convergence_reason,
        "overall_slot_accuracy_improvement": tracker.get_improvement("overall_slot_accuracy"),
        "exact_match_rate_improvement": tracker.get_improvement("exact_match_rate"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report_dict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
