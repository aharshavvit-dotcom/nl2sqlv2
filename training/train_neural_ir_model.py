"""Train the Neural QueryIR Model.

This is a compatibility wrapper around the optimized trainer.
Prefer ``python training/train_model.py --config configs/training.yaml``
for full integrated training.

Usage:
    python training/train_neural_ir_model.py --train data/processed/generic_ir_train.jsonl --validation data/processed/generic_ir_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _resolve_dir(new_name: str, old_name: str) -> Path:
    new_path = ROOT / "artifacts" / new_name
    return new_path if new_path.exists() else ROOT / "artifacts" / old_name


def main() -> None:
    import warnings
    warnings.warn(
        "This command is a compatibility wrapper. "
        "Prefer `python training/train_model.py --config configs/training.yaml` for full integrated training.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("WARNING: This command is a compatibility wrapper. "
          "Prefer `python training/train_model.py --config configs/training.yaml` for full integrated training.")

    parser = argparse.ArgumentParser(description="Train the Neural QueryIR Model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train", type=str, default=str(ROOT / "data" / "processed" / "generic_ir_train.jsonl"), help="Path to generic IR training JSONL")
    parser.add_argument("--validation", type=str, default=str(ROOT / "data" / "processed" / "generic_ir_validation.jsonl"), help="Path to generic IR validation JSONL")
    parser.add_argument("--hard-negatives", type=str, default=None, help="Path to hard-negative JSONL")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "work" / "neural_ir"), help="Artifact output directory")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Artifact output directory")
    parser.add_argument("--training-data", type=str, default=None,
                        help="Path to IR training corpus JSONL")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dataset-filter", type=str, default=None)
    parser.add_argument("--intent-filter", type=str, default=None)
    parser.add_argument("--complexity-filter", type=str, default=None)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--optimized", action="store_true",
                        help="Deprecated no-op; optimized training is now the default.")
    parser.add_argument("--legacy", action="store_true",
                        help="Explicitly use the old legacy trainer path.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "neural_training_default.yaml"),
                        help="YAML config path (used with --optimized)")
    args = parser.parse_args()

    # Delegate to optimized trainer by default. Legacy mode must be explicit.
    if not args.legacy:
        from training.train_neural_ir_optimized import main as optimized_main
        new_argv = [sys.argv[0]]
        if args.config:
            new_argv.extend(["--config", args.config])
        if args.train:
            new_argv.extend(["--train", args.train])
        if args.validation:
            new_argv.extend(["--validation", args.validation])
        if args.hard_negatives:
            new_argv.extend(["--hard-negatives", args.hard_negatives])
        if args.output_dir or args.artifact_dir:
            new_argv.extend(["--output-dir", args.output_dir or args.artifact_dir])
        if args.epochs:
            new_argv.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            new_argv.extend(["--batch-size", str(args.batch_size)])
        if args.max_examples:
            new_argv.extend(["--max-examples", str(args.max_examples)])
        sys.argv = new_argv
        optimized_main()
        return

    artifact_dir = Path(args.output_dir or args.artifact_dir) if (args.output_dir or args.artifact_dir) else _resolve_dir("neural_ir_model", "option_a_ir_model_v2")

    if args.train:
        train_path = Path(args.train)
        validation_path = Path(args.validation) if args.validation else train_path.parent / "generic_ir_validation.jsonl"
    elif args.training_data:
        train_path = Path(args.training_data)
        validation_name = train_path.name.replace("training", "validation").replace("train", "validation")
        validation_path = train_path.parent / validation_name
        if not validation_path.exists():
            validation_path = train_path
    else:
        print("Error: --legacy requires --train or --training-data. Default training uses generic_ir_train.jsonl.")
        sys.exit(1)

    if not train_path.exists():
        print(f"Error: Training data file not found at {train_path}")
        print("Please build the QueryIR training data first by running:")
        print("  python training/build_generic_ir_corpus.py --datasets wikisql,spider,bird-mini --output-dir data/processed")
        sys.exit(1)

    from training_ir.train_option_a_v2_model import train_option_a_v2_model
    filtered_train, filtered_validation = _maybe_filter_inputs(train_path, validation_path, artifact_dir, args)
    report = train_option_a_v2_model(
        train_path=filtered_train,
        validation_path=filtered_validation,
        hard_negatives_path=Path(args.hard_negatives) if args.hard_negatives else None,
        output_dir=artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


def _maybe_filter_inputs(train_path: Path, validation_path: Path, artifact_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    filters = {
        "dataset_name": _csv(args.dataset_filter),
        "intent": _csv(args.intent_filter),
        "complexity": _csv(args.complexity_filter),
    }
    if not any(filters.values()) and not args.curriculum:
        return train_path, validation_path

    artifact_dir.mkdir(parents=True, exist_ok=True)
    filtered_train = artifact_dir / "_filtered_generic_train.jsonl"
    filtered_validation = artifact_dir / "_filtered_generic_validation.jsonl"
    _write_jsonl(filtered_train, _filter_rows(_read_jsonl(train_path), filters, curriculum=args.curriculum))
    _write_jsonl(filtered_validation, _filter_rows(_read_jsonl(validation_path), filters, curriculum=args.curriculum))
    return filtered_train, filtered_validation


def _filter_rows(rows: list[dict[str, Any]], filters: dict[str, set[str]], curriculum: bool) -> list[dict[str, Any]]:
    filtered = []
    allowed_curriculum = {
        "show_records",
        "count_records",
        "simple_filter",
        "metric_summary",
        "metric_by_dimension",
        "count_by_dimension",
        "top_n_metric_by_dimension",
        "bottom_n_metric_by_dimension",
        "trend_by_date",
    }
    for row in rows:
        if filters["dataset_name"] and str(row.get("dataset_name")) not in filters["dataset_name"]:
            continue
        intent = str(row.get("intent") or (row.get("query_ir") or {}).get("intent") or "")
        if filters["intent"] and intent not in filters["intent"]:
            continue
        if filters["complexity"] and str(row.get("complexity")) not in filters["complexity"]:
            continue
        if curriculum and intent not in allowed_curriculum:
            continue
        filtered.append(row)
    return filtered


def _csv(value: str | None) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
