from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.ablation import OptionAAblationEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run available Option A ablation comparisons.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = OptionAAblationEvaluator().run(str(args.test), db_path=str(args.db) if args.db else None, output_path=str(args.output))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
