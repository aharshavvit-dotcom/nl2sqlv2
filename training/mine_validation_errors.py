from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl, write_json, write_jsonl
from self_training.hard_negative_miner import HardNegativeMiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negatives from validation prediction mistakes.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "self_training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}")
    result = HardNegativeMiner().mine(read_jsonl(args.predictions))
    write_jsonl(args.output_dir / "mined_hard_negatives.jsonl", result["mined_hard_negatives"])
    write_json(args.output_dir / "error_summary.json", result["error_summary"])
    print(json.dumps(result["error_summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
