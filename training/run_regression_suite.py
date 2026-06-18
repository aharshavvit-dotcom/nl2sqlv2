from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.reporting import save_report_pair
from dataset_training.utils import read_jsonl
from quality_gates.regression_suite import DEFAULT_CASES, RegressionSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic NL-to-SQL regression cases.")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "generic_benchmark_cases.jsonl")
    parser.add_argument("--feedback-regressions", type=Path, default=ROOT / "data" / "processed" / "feedback_safety_regressions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "regression_suite_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = read_jsonl(args.cases) if args.cases.exists() else DEFAULT_CASES
    feedback_regressions = read_jsonl(args.feedback_regressions)
    report = RegressionSuite().run(cases=cases, feedback_safety_regressions=feedback_regressions)
    save_report_pair(args.output, report, "Regression Suite Report")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
