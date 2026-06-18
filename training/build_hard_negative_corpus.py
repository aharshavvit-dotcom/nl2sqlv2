from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.hard_negative_corpus_builder import HardNegativeCorpusBuilder
from dataset_training.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generic hard-negative QueryIR corpus.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "generic_ir_train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "generic_ir_hard_negatives.jsonl")
    parser.add_argument("--max-negatives-per-example", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    negatives = HardNegativeCorpusBuilder().build(rows, max_negatives_per_example=args.max_negatives_per_example)
    write_jsonl(args.output, negatives)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "negative_count": len(negatives)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
