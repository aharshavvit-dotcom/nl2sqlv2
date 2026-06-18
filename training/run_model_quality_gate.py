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
    report = ModelQualityGate().evaluate(evaluation_report, load_thresholds(args.thresholds))
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
