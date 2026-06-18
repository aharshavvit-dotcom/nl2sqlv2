from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import write_json
from quality_gates.release_checker import ReleaseChecker


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release readiness checks for the generic NL-to-SQL stack.")
    parser.add_argument("--audit-report", type=Path, default=ROOT / "artifacts" / "audit" / "generic_nl2sql_readiness_report.json")
    parser.add_argument("--evaluation-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--quality-gate-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "model_quality_gate_report.json")
    parser.add_argument("--regression-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "regression_suite_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "release_readiness_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = ReleaseChecker().evaluate(
        audit_report=_read_json(args.audit_report),
        evaluation_report=_read_json(args.evaluation_report),
        quality_gate_report=_read_json(args.quality_gate_report),
        regression_report=_read_json(args.regression_report),
        repo_root=ROOT,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
