from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl
from self_training.ranking_trainer import RankingTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive ranker from candidate-vs-gold comparisons.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "adaptive_ranker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}")
    report = RankingTrainer().train(read_jsonl(args.predictions), args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
