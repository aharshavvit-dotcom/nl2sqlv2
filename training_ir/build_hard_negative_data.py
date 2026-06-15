from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.hard_negative_builder import HardNegativeBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-negative QueryIR rows for Option A V2.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-negatives-per-example", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = HardNegativeBuilder().build_file(
        input_path=str(args.input),
        output_path=str(args.output),
        max_negatives_per_example=args.max_negatives_per_example,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
