"""CLI: Run the complete dataset-driven self-improvement training loop.

Usage
-----
python training/run_self_training_loop.py \\
  --train data/processed/generic_ir_train.jsonl \\
  --validation data/processed/generic_ir_validation.jsonl \\
  --test data/processed/generic_ir_test.jsonl \\
  --output-dir artifacts/self_training \\
  --max-iterations 3 \\
  --epochs-per-iteration 10 \\
  --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_training.self_training_loop import SelfTrainingConfig, SelfTrainingLoop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dataset-driven self-improvement training loop.",
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "processed" / "generic_ir_train.jsonl",
                        help="Path to training JSONL file.")
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "processed" / "generic_ir_validation.jsonl",
                        help="Path to validation JSONL file.")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "generic_ir_test.jsonl",
                        help="Path to test JSONL file.")
    parser.add_argument("--model-output-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model",
                        help="Directory to save the final model.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "self_training",
                        help="Directory for self-training artifacts.")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum self-improvement iterations (default: 3).")
    parser.add_argument("--min-improvement", type=float, default=0.005,
                        help="Minimum improvement threshold to continue (default: 0.005).")
    parser.add_argument("--correction-weight", type=float, default=2.0,
                        help="Weight for correction examples (default: 2.0).")
    parser.add_argument("--hard-negative-weight", type=float, default=1.5,
                        help="Weight for hard negative examples (default: 1.5).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size per iteration (default: 32).")
    parser.add_argument("--epochs-per-iteration", type=int, default=10,
                        help="Training epochs per iteration (default: 10).")
    parser.add_argument("--no-hard-negatives", action="store_true",
                        help="Disable hard negative generation.")
    parser.add_argument("--no-corrections", action="store_true",
                        help="Disable correction example generation.")
    parser.add_argument("--max-prediction-examples", type=int, default=None,
                        help="Limit prediction examples per iteration (for quick iteration).")
    parser.add_argument("--optimized-neural-training", action="store_true",
                        help="Use the optimized neural training loop with configurable optimizer/scheduler.")
    parser.add_argument("--neural-config", type=str, default=None,
                        help="YAML config path for optimized neural training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = SelfTrainingConfig(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        model_output_dir=args.model_output_dir,
        artifacts_dir=args.output_dir,
        max_iterations=args.max_iterations,
        min_improvement=args.min_improvement,
        correction_weight=args.correction_weight,
        hard_negative_weight=args.hard_negative_weight,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs_per_iteration,
        use_hard_negatives=not args.no_hard_negatives,
        use_corrections=not args.no_corrections,
        max_prediction_examples=args.max_prediction_examples,
        use_optimized_training=args.optimized_neural_training,
        neural_config_path=args.neural_config,
    )

    loop = SelfTrainingLoop(config)
    report = loop.run()

    print("\n" + json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
