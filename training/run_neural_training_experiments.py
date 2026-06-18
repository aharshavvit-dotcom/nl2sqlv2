"""CLI for running neural training experiment grid.

Usage:
    python training/run_neural_training_experiments.py \\
      --grid configs/neural_experiment_grid.yaml \\
      --train data/processed/generic_ir_train.jsonl \\
      --validation data/processed/generic_ir_validation.jsonl \\
      --output-dir artifacts/neural_experiments \\
      --max-examples 1000 \\
      --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_optimization.training_config import NeuralTrainingConfig, merge_cli_overrides
from neural_optimization.experiment_runner import ExperimentRunner, load_experiment_grid
from neural_optimization.experiment_reporter import ExperimentReporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural training experiment grid")
    parser.add_argument("--grid", type=str, required=True, help="Experiment grid YAML")
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--validation", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/neural_experiments")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    # Build base config with CLI overrides
    base = NeuralTrainingConfig()
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_examples": args.max_examples,
        "train": args.train,
        "validation": args.validation,
        "output_dir": args.output_dir,
    }
    base = merge_cli_overrides(base, overrides)

    grid = load_experiment_grid(args.grid)
    output_dir = Path(args.output_dir)

    runner = ExperimentRunner(base, grid, output_dir)

    def train_fn(config: NeuralTrainingConfig, exp_dir: Path) -> dict:
        from training.train_neural_ir_optimized import run_optimized_training
        config.output["output_dir"] = str(exp_dir)
        return run_optimized_training(config, exp_dir)

    results = runner.run(train_fn)
    reporter = ExperimentReporter(results)
    reporter.save(output_dir)
    print(f"\nExperiment summary written to {output_dir / 'experiment_summary.md'}")

    best = reporter.best_experiment()
    if best:
        print(f"Best experiment: {best['name']} (slot accuracy: "
              f"{(best.get('metrics') or {}).get('overall_slot_accuracy', 0):.4f})")


if __name__ == "__main__":
    main()
