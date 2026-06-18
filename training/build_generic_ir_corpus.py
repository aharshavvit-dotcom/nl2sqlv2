from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training import DatasetRegistry, DatasetSplitManager, GenericIRCorpusBuilder
from scripts.dataset_paths import parse_dataset_list


def build_generic_ir_corpus(args: argparse.Namespace) -> dict:
    builder = GenericIRCorpusBuilder(
        dataset_registry=DatasetRegistry(),
        split_manager=DatasetSplitManager(
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            unseen_db_test_ratio=args.unseen_db_test_ratio,
        ),
        sql_to_ir_converter=None,
        quality_filter=None,
    )
    return builder.build(
        datasets=parse_dataset_list(args.datasets),
        max_examples=args.max_examples if args.max_examples and args.max_examples > 0 else None,
        output_dir=str(args.output_dir),
        artifact_dir=str(args.artifact_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generic dataset-scale QueryIR corpus.")
    parser.add_argument("--datasets", default="wikisql,spider,bird-mini")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "generic_training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--unseen-db-test-ratio", type=float, default=0.15)
    parser.add_argument("--include-unsupported", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    report = build_generic_ir_corpus(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
