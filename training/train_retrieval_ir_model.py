"""Train the Retrieval QueryIR Model from local datasets.

This is the canonical entry point for training the retrieval model.
It wraps ``training.train_retriever_from_datasets.train_from_datasets``.

Usage:
    python training/train_retrieval_ir_model.py [--datasets wikisql spider bird-mini] [--max-examples 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _resolve_dir(new_name: str, old_name: str) -> Path:
    new_path = ROOT / "artifacts" / new_name
    return new_path if new_path.exists() else ROOT / "artifacts" / old_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Retrieval QueryIR Model")
    parser.add_argument("--datasets", nargs="+", default=["wikisql", "spider", "bird-mini"],
                        help="Datasets to train from (default: wikisql spider bird-mini)")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max examples per dataset (0 = no limit)")
    parser.add_argument("--include-schema-text", action="store_true",
                        help="Include schema text in training examples")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Artifact output directory")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else _resolve_dir("retrieval_ir_model", "option_c_model")

    from training.train_retriever_from_datasets import train_from_datasets
    report = train_from_datasets(
        args.datasets,
        artifact_dir=artifact_dir,
        max_examples=args.max_examples or None,
        include_schema_text=args.include_schema_text,
    )
    print("Training complete.")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
