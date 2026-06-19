from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import write_json
from quality_gates.model_quality_gate import ModelQualityGate
from quality_gates.thresholds import load_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model metrics against production quality thresholds.")
    parser.add_argument("--evaluation-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evaluation" / "model_quality_thresholds.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "model_quality_gate_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_report = json.loads(args.evaluation_report.read_text(encoding="utf-8")) if args.evaluation_report.exists() else {}
    contribution_path = ROOT / "artifacts" / "generic_training" / "dataset_contribution_report.json"
    if contribution_path.exists():
        evaluation_report["dataset_contribution_report_required"] = True
        evaluation_report["dataset_contribution_report"] = json.loads(contribution_path.read_text(encoding="utf-8"))
    execution_path = args.evaluation_report.parent / "execution_aware_evaluation_report.json"
    if execution_path.exists():
        execution_report = json.loads(execution_path.read_text(encoding="utf-8"))
        summary = execution_report.get("summary") or {}
        if "execution_match_rate" in summary:
            evaluation_report.setdefault("summary", {})["execution_match_rate"] = summary["execution_match_rate"]
        evaluation_report["execution_aware_evaluation"] = {**execution_report, "enabled": True, "required": False}
    else:
        evaluation_report["execution_aware_evaluation"] = {
            "enabled": False,
            "required": False,
            "reason": "disabled by config",
        }
    report = ModelQualityGate().evaluate(evaluation_report, load_thresholds(args.thresholds))
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
