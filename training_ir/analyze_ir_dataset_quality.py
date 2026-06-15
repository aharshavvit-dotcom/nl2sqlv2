from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.dataset_quality import IRDatasetQualityAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QueryIR training dataset quality.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--unsupported", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = IRDatasetQualityAnalyzer().analyze(str(args.input))
    if args.unsupported and args.unsupported.exists():
        unsupported_report = IRDatasetQualityAnalyzer().analyze(str(args.unsupported))
        report["unsupported_rows"] = unsupported_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
