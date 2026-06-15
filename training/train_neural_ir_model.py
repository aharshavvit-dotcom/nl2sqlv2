"""Train the Neural QueryIR Model.

This is the canonical entry point for training the neural IR model.
It wraps ``training_ir.train_option_a_v2_model``.

Usage:
    python training/train_neural_ir_model.py [--epochs 30] [--batch-size 32]
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
    parser = argparse.ArgumentParser(description="Train the Neural QueryIR Model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Artifact output directory")
    parser.add_argument("--training-data", type=str, default=None,
                        help="Path to IR training corpus JSONL")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else _resolve_dir("neural_ir_model", "option_a_ir_model_v2")

    if args.training_data:
        train_path = Path(args.training_data)
        validation_name = train_path.name.replace("training", "validation").replace("train", "validation")
        validation_path = train_path.parent / validation_name
        if not validation_path.exists():
            validation_path = train_path
    else:
        # Fall back to default location
        train_path = ROOT / "data" / "processed" / "ir_training_examples.jsonl"
        validation_path = ROOT / "data" / "processed" / "ir_validation_examples.jsonl"

    if not train_path.exists():
        print(f"Error: Training data file not found at {train_path}")
        print("Please build the QueryIR training data first by running:")
        print("  python training_ir/build_ir_training_data.py --datasets wikisql,spider,bird-mini --output-dir training_data")
        sys.exit(1)

    from training_ir.train_option_a_v2_model import train_option_a_v2_model
    train_option_a_v2_model(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
