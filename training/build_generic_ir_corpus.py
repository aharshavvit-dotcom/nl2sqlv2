from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
            model_selection_ratio=getattr(args, "model_selection_ratio", 0.0),
            test_ratio=args.test_ratio,
            unseen_db_test_ratio=args.unseen_db_test_ratio,
            split_version=getattr(args, "split_version", "semantic_v1"),
            force_create_new_version=bool(getattr(args, "force_create_new_version", False)),
            parent_split_version=getattr(args, "parent_split_version", None),
            regeneration_reason=getattr(args, "split_regeneration_reason", None),
        ),
        sql_to_ir_converter=None,
        quality_filter=None,
    )
    return builder.build(
        datasets=parse_dataset_list(args.datasets),
        max_examples=args.max_examples if args.max_examples and args.max_examples > 0 else None,
        output_dir=str(args.output_dir),
        artifact_dir=str(args.artifact_dir),
        max_examples_per_dataset=_parse_int_map(getattr(args, "max_examples_per_dataset", None)),
        min_converted_examples_required=_parse_int_map(getattr(args, "min_converted_examples_required", None)),
        schema_renaming=getattr(args, "schema_renaming", None),
        pipeline_run_id=getattr(args, "pipeline_run_id", None),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generic dataset-scale QueryIR corpus.")
    parser.add_argument("--datasets", default="wikisql,spider,bird-mini")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument(
        "--max-examples-per-dataset",
        default=None,
        help="Comma-separated per-dataset caps, e.g. wikisql=5000,spider=5000,bird-mini=5000",
    )
    parser.add_argument(
        "--min-converted-examples-required",
        default=None,
        help="Comma-separated minimum usable QueryIR counts, e.g. wikisql=100,spider=100,bird-mini=100",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "generic_training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--model-selection-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--unseen-db-test-ratio", type=float, default=0.15)
    parser.add_argument("--split-version", default="semantic_v1")
    parser.add_argument("--force-create-new-version", action="store_true", default=False)
    parser.add_argument("--parent-split-version", default=None)
    parser.add_argument("--split-regeneration-reason", default=None)
    parser.add_argument("--include-unsupported", action="store_true", default=True)
    return parser.parse_args()


def _parse_int_map(value: Any) -> dict[str, int] | None:
    if not value:
        return None
    if isinstance(value, dict):
        return {str(key): int(item) for key, item in value.items() if item is not None}
    result: dict[str, int] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value in per-dataset map: {item}")
        key, raw = item.split("=", 1)
        result[key.strip()] = int(raw.strip())
    return result


def main() -> int:
    report = build_generic_ir_corpus(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
