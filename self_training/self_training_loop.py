"""Self-Training Loop — the main orchestrator for dataset-driven self-improvement.

Executes the complete loop:
  1. Load gold training / validation / test splits
  2. Train baseline model (or load existing)
  3. Loop for ``max_iterations``:
     a. Predict on validation set
     b. Compare predictions vs gold
     c. Classify errors
     d. Generate hard negatives from errors
     e. Generate correction examples
     f. Augment training set
     g. Retrain model on augmented set
     h. Evaluate on validation set
     i. Record metrics
     j. Check convergence
  4. Evaluate best model on test set
  5. Return improvement report
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .correction_generator import CorrectionExampleGenerator
from .error_classifier import ErrorClassifier
from .gold_comparator import GoldComparator
from .hard_negative_generator import PredictionHardNegativeGenerator
from .improvement_tracker import ImprovementTracker
from .model_selector import ModelSelector
from .prediction_runner import PredictionRunner


@dataclass
class SelfTrainingConfig:
    """Configuration for the self-training loop."""

    train_path: Path = Path("data/processed/generic_ir_train.jsonl")
    validation_path: Path = Path("data/processed/generic_ir_validation.jsonl")
    test_path: Path = Path("data/processed/generic_ir_test.jsonl")
    model_output_dir: Path = Path("artifacts/neural_ir_model")
    artifacts_dir: Path = Path("artifacts/self_training")
    max_iterations: int = 3
    min_improvement: float = 0.005
    correction_weight: float = 2.0
    hard_negative_weight: float = 1.5
    batch_size: int = 32
    epochs_per_iteration: int = 10
    use_hard_negatives: bool = True
    use_corrections: bool = True
    max_prediction_examples: int | None = None
    use_optimized_training: bool = False
    neural_config_path: str | None = None



class SelfTrainingLoop:
    """Orchestrates the complete self-improvement loop."""

    def __init__(self, config: SelfTrainingConfig):
        self.config = config
        self.comparator = GoldComparator()
        self.classifier = ErrorClassifier()
        self.neg_generator = PredictionHardNegativeGenerator()
        self.correction_generator = CorrectionExampleGenerator(correction_weight=config.correction_weight)
        self.tracker = ImprovementTracker(config.artifacts_dir)
        self.model_selector = ModelSelector()

    def run(self) -> dict[str, Any]:
        """Execute the full self-improvement loop.

        Returns the improvement report as a dict.
        """

        config = self.config
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("SELF-TRAINING LOOP — Dataset-Driven Self-Improvement")
        print("=" * 70)

        # --- Load data ---
        train_data = _read_jsonl(config.train_path)
        val_data = _read_jsonl(config.validation_path)
        test_data = _read_jsonl(config.test_path)

        print(f"Training examples:   {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        print(f"Test examples:       {len(test_data)}")

        if not train_data:
            print("WARNING: No training data found.  Skipping self-training loop.")
            return self.tracker.generate_report().to_dict()

        # --- Iteration 0: Baseline evaluation ---
        print("\n" + "=" * 70)
        print("Iteration 0: Baseline evaluation")
        print("=" * 70)

        iteration_dir = config.artifacts_dir / "iteration_0"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        if (config.model_output_dir / "model.pt").exists():
            print(f"Loading existing baseline model from {config.model_output_dir}")
            baseline_metrics = self._evaluate_iteration(
                model_dir=config.model_output_dir,
                val_data=val_data,
                iteration=0,
                iteration_dir=iteration_dir,
            )
        else:
            print("No baseline model found. Training from scratch...")
            self._train_model(train_data, config.model_output_dir, iteration=0)
            baseline_metrics = self._evaluate_iteration(
                model_dir=config.model_output_dir,
                val_data=val_data,
                iteration=0,
                iteration_dir=iteration_dir,
            )

        self.tracker.record_iteration(0, baseline_metrics)
        _save_json(iteration_dir / "metrics.json", baseline_metrics)

        # --- Self-improvement iterations ---
        current_train = list(train_data)
        current_model_dir = config.model_output_dir

        for iteration in range(1, config.max_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"Iteration {iteration}/{config.max_iterations}: Self-Improvement")
            print(f"{'=' * 70}")

            iter_start = time.time()
            iteration_dir = config.artifacts_dir / f"iteration_{iteration}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Predict on validation set
            print(f"\nStep 1/{6}: Predicting on validation set...")
            try:
                runner = PredictionRunner(current_model_dir)
                predictions = runner.predict_batch(val_data, max_examples=config.max_prediction_examples)
            except Exception as exc:
                print(f"  WARNING: Prediction failed: {exc}")
                print(f"  Skipping iteration {iteration} (no usable model)")
                self.tracker.record_iteration(iteration, {"overall_slot_accuracy": 0.0, "exact_match_rate": 0.0})
                break
            _save_jsonl(iteration_dir / "predictions.jsonl", predictions)
            print(f"  → {len(predictions)} predictions generated")

            # Step 2: Compare predictions vs gold
            print(f"Step 2/{6}: Comparing predictions vs gold...")
            comparison = self.comparator.compare_batch(predictions, val_data)
            _save_json(iteration_dir / "comparison_report.json", {
                "total": comparison.total,
                "exact_matches": comparison.exact_matches,
                "partial_matches": comparison.partial_matches,
                "failures": comparison.failures,
                "field_accuracy": comparison.field_accuracy,
            })
            print(f"  → Exact: {comparison.exact_matches} | Partial: {comparison.partial_matches} | Failures: {comparison.failures}")

            # Step 3: Classify errors
            print(f"Step 3/{6}: Classifying errors...")
            error_report = self.classifier.classify_batch(comparison.per_example, predictions)
            _save_json(iteration_dir / "error_report.json", {
                "total_errors": error_report.total_errors,
                "by_category": error_report.by_category,
                "by_severity": error_report.by_severity,
                "top_error_categories": error_report.top_error_categories,
            })
            print(f"  → {error_report.total_errors} errors classified")
            for cat, count in error_report.top_error_categories[:5]:
                print(f"    - {cat}: {count}")

            # Step 4: Generate hard negatives + corrections
            print(f"Step 4/{6}: Generating hard negatives and corrections...")
            hard_negatives: list[dict[str, Any]] = []
            corrections: list[dict[str, Any]] = []

            if config.use_hard_negatives:
                hard_negatives = self.neg_generator.generate_from_errors(
                    error_report.classifications, predictions
                )
                _save_jsonl(iteration_dir / "hard_negatives.jsonl", hard_negatives)
                print(f"  → {len(hard_negatives)} hard negatives generated")

            if config.use_corrections:
                corrections = self.correction_generator.generate(
                    error_report.classifications, predictions
                )
                _save_jsonl(iteration_dir / "corrections.jsonl", corrections)
                print(f"  → {len(corrections)} correction examples generated")

            # Step 5: Augment training set and retrain
            print(f"Step 5/{6}: Augmenting training set and retraining...")
            augmented = self.correction_generator.generate_augmented_training_set(
                current_train, corrections, hard_negatives,
                hard_negative_weight=config.hard_negative_weight,
            )
            print(f"  → Augmented training set: {len(augmented)} examples (was {len(current_train)})")

            iter_model_dir = iteration_dir / "model"
            self._train_model(augmented, iter_model_dir, iteration=iteration)
            current_model_dir = iter_model_dir

            # Step 6: Evaluate
            print(f"Step 6/{6}: Evaluating iteration {iteration}...")
            iter_metrics = self._evaluate_iteration(
                model_dir=iter_model_dir,
                val_data=val_data,
                iteration=iteration,
                iteration_dir=iteration_dir,
            )
            self.tracker.record_iteration(iteration, iter_metrics)
            _save_json(iteration_dir / "metrics.json", iter_metrics)

            iter_duration = time.time() - iter_start
            print(f"\nIteration {iteration} completed in {iter_duration:.1f}s")
            print(f"  overall_slot_accuracy: {iter_metrics.get('overall_slot_accuracy', 0):.4f}")
            print(f"  exact_match_rate: {iter_metrics.get('exact_match_rate', 0):.4f}")

            # Check convergence
            if self.tracker.should_stop(min_improvement=config.min_improvement):
                print(f"\nConverged: improvement < {config.min_improvement:.4f}. Stopping.")
                break

            # Use augmented set for next iteration
            current_train = augmented

        # --- Final: Evaluate best model on test set ---
        print(f"\n{'=' * 70}")
        print("Final: Selecting best model and evaluating on test set")
        print(f"{'=' * 70}")

        report = self.tracker.generate_report()
        best_iter = report.best_iteration
        best_model_dir = (
            config.model_output_dir if best_iter == 0
            else config.artifacts_dir / f"iteration_{best_iter}" / "model"
        )

        # Promote best model
        if best_iter > 0:
            promo = self.model_selector.promote_best(best_model_dir, config.model_output_dir)
            print(f"  Promoted iteration {best_iter} model: {promo.get('files_copied', [])}")

        # Final test evaluation
        if test_data and (config.model_output_dir / "model.pt").exists():
            test_metrics = self._evaluate_iteration(
                model_dir=config.model_output_dir,
                val_data=test_data,
                iteration=-1,
                iteration_dir=config.artifacts_dir / "final_test",
            )
            _save_json(config.artifacts_dir / "final_test_metrics.json", test_metrics)
            print(f"  Test set overall_slot_accuracy: {test_metrics.get('overall_slot_accuracy', 0):.4f}")
        else:
            test_metrics = {}

        # Save final report
        report_dict = report.to_dict()
        report_dict["test_metrics"] = test_metrics
        _save_json(config.artifacts_dir / "improvement_report.json", report_dict)

        print(f"\nSelf-training complete. Best iteration: {best_iter}")
        print(f"Report saved to: {config.artifacts_dir / 'improvement_report.json'}")

        return report_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_model(
        self,
        train_data: list[dict[str, Any]],
        output_dir: Path,
        iteration: int,
    ) -> None:
        """Train (or retrain) the neural IR model on training data.

        This delegates to the existing training pipeline.
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save training data as JSONL for the trainer to consume
        train_path = output_dir / "train_data.jsonl"
        _save_jsonl(train_path, train_data)

        print(f"  Training model (iteration {iteration})...")
        print(f"  Output: {output_dir}")
        print(f"  Examples: {len(train_data)} | Epochs: {self.config.epochs_per_iteration} | Batch: {self.config.batch_size}")

        try:
            from training_ir.train_option_a_v2_model import train_option_a_v2

            train_option_a_v2(
                train_path=str(train_path),
                output_dir=str(output_dir),
                epochs=self.config.epochs_per_iteration,
                batch_size=self.config.batch_size,
                use_hard_negative_loss=self.config.use_hard_negatives,
            )
            print(f"  Model training complete for iteration {iteration}")
        except Exception as exc:
            print(f"  WARNING: Training failed for iteration {iteration}: {exc}")
            print(f"  Continuing with previous model...")

    def _evaluate_iteration(
        self,
        model_dir: Path,
        val_data: list[dict[str, Any]],
        iteration: int,
        iteration_dir: Path,
    ) -> dict[str, Any]:
        """Evaluate a model on validation data and return metrics.

        Uses the GoldComparator for detailed field-level metrics.
        """

        iteration_dir.mkdir(parents=True, exist_ok=True)

        if not (model_dir / "model.pt").exists():
            print(f"  No model.pt found at {model_dir}; returning zero metrics")
            return {"overall_slot_accuracy": 0.0, "exact_match_rate": 0.0}

        # Run predictions
        runner = PredictionRunner(model_dir)
        predictions = runner.predict_batch(val_data, max_examples=self.config.max_prediction_examples)

        # Compare against gold
        comparison = self.comparator.compare_batch(predictions, val_data)

        # Build metrics dict
        metrics: dict[str, Any] = {
            "iteration": iteration,
            "total_examples": comparison.total,
            "exact_matches": comparison.exact_matches,
            "partial_matches": comparison.partial_matches,
            "exact_match_rate": comparison.exact_matches / max(comparison.total, 1),
            "partial_match_rate": comparison.partial_matches / max(comparison.total, 1),
            "match_score_mean": (
                sum(r.match_score for r in comparison.per_example) / max(len(comparison.per_example), 1)
            ),
        }

        # Map field accuracy to tracked metric names
        field_metric_map = {
            "intent": "intent_accuracy",
            "base_table": "base_table_accuracy",
            "metrics": "metric_accuracy",
            "dimensions": "dimension_accuracy",
            "filters": "filter_accuracy",
            "date_filters": "date_filter_accuracy",
            "joins": "join_accuracy",
            "order_by": "order_accuracy",
            "limit": "limit_accuracy",
        }
        for field_name, metric_name in field_metric_map.items():
            metrics[metric_name] = comparison.field_accuracy.get(field_name, 0.0)

        # Compute overall slot accuracy as the mean of field accuracies
        slot_fields = ["intent", "base_table", "metrics", "dimensions", "filters", "date_filters"]
        slot_values = [comparison.field_accuracy.get(f, 0.0) for f in slot_fields]
        metrics["overall_slot_accuracy"] = sum(slot_values) / max(len(slot_values), 1)

        # SQL validation rate from predictions
        sql_valid_count = sum(
            1 for p in predictions
            if (p.get("sql_validation") or {}).get("is_valid",
                (p.get("sql_validation") or {}).get("ok", False))
        )
        metrics["sql_validation_rate"] = sql_valid_count / max(len(predictions), 1)

        return metrics


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _save_json(path: Path, data: Any) -> None:
    """Write a dict/list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
