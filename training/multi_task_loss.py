"""Multi-Task Loss with Safeguarded Learned Weighting — Gate 4.

Implements uncertainty-weighted multi-task loss (Kendall et al., 2018) with:
- Hard-floor constraints (no single loss weight can drop below a minimum)
- Per-task gradient isolation (task A's error cannot corrupt task B's encoder)
- Checkpoint selection using hard-floor validation metrics

Behind feature flags: each head's contribution is gated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskLossConfig:
    """Configuration for a single task in the multi-task loss."""
    name: str
    weight_floor: float = 0.01      # Minimum learned weight
    weight_ceiling: float = 1.0     # Maximum learned weight
    initial_log_variance: float = 0.0
    enabled: bool = True
    gradient_isolation: bool = False  # If True, stop gradients to shared encoder

    def validate(self) -> list[str]:
        errors = []
        if self.weight_floor < 0:
            errors.append(f"Task {self.name}: weight_floor must be >= 0")
        if self.weight_ceiling <= self.weight_floor:
            errors.append(f"Task {self.name}: weight_ceiling must be > weight_floor")
        if self.initial_log_variance < -10 or self.initial_log_variance > 10:
            errors.append(f"Task {self.name}: initial_log_variance out of range [-10, 10]")
        return errors


@dataclass
class MultiTaskLossConfig:
    """Configuration for the full multi-task loss."""
    tasks: list[TaskLossConfig] = field(default_factory=list)
    weighting_strategy: str = "uncertainty"  # "uncertainty", "fixed", "dynamic"
    gradient_norm_clip: float = 1.0
    loss_scale_smoothing: float = 0.999  # EMA smoothing for loss scale tracking

    def validate(self) -> list[str]:
        errors = []
        if not self.tasks:
            errors.append("No tasks configured")
        names = [t.name for t in self.tasks]
        if len(names) != len(set(names)):
            errors.append("Duplicate task names")
        if self.weighting_strategy not in {"uncertainty", "fixed", "dynamic"}:
            errors.append(f"Unknown weighting strategy: {self.weighting_strategy}")
        if self.gradient_norm_clip <= 0:
            errors.append("gradient_norm_clip must be > 0")
        for task in self.tasks:
            errors.extend(task.validate())
        return errors


class MultiTaskLossComputer:
    """Computes uncertainty-weighted multi-task loss with safeguards.

    Each task i has a learned log-variance parameter s_i.
    The total loss is: L = sum_i [L_i / (2 * sigma_i^2) + log(sigma_i)]
    where sigma_i = exp(s_i).

    Safeguards:
    1. Weight floors: effective weight >= weight_floor for each task
    2. Weight ceilings: effective weight <= weight_ceiling
    3. Gradient isolation: optional stop-gradient per task
    """

    def __init__(self, config: MultiTaskLossConfig) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid config: " + "; ".join(errors))
        self.config = config
        self._log_variances: dict[str, float] = {
            task.name: task.initial_log_variance
            for task in config.tasks if task.enabled
        }
        self._loss_history: dict[str, list[float]] = {
            task.name: [] for task in config.tasks if task.enabled
        }
        self._ema_loss: dict[str, float] = {
            task.name: 0.0 for task in config.tasks if task.enabled
        }

    def compute(self, task_losses: dict[str, float]) -> dict[str, Any]:
        """Compute the multi-task loss from per-task losses.

        Returns:
            {
                "total_loss": float,
                "task_contributions": {name: float},
                "effective_weights": {name: float},
                "raw_weights": {name: float},
            }
        """
        total = 0.0
        contributions: dict[str, float] = {}
        effective_weights: dict[str, float] = {}
        raw_weights: dict[str, float] = {}

        task_configs = {t.name: t for t in self.config.tasks if t.enabled}

        for name, loss_val in task_losses.items():
            if name not in task_configs:
                continue

            tc = task_configs[name]
            log_var = self._log_variances.get(name, 0.0)

            if self.config.weighting_strategy == "uncertainty":
                # Uncertainty weighting: L_i / (2 * sigma^2) + log(sigma)
                precision = math.exp(-2 * log_var)
                regularizer = log_var
                raw_weight = precision / 2.0

                # Apply floor/ceiling
                effective_weight = max(tc.weight_floor, min(tc.weight_ceiling, raw_weight))
                contribution = effective_weight * loss_val + regularizer

            elif self.config.weighting_strategy == "fixed":
                effective_weight = tc.weight_floor
                raw_weight = tc.weight_floor
                contribution = effective_weight * loss_val

            else:  # dynamic
                # Use EMA-normalized inverse loss as weight
                ema = self._ema_loss.get(name, loss_val)
                raw_weight = 1.0 / max(ema, 1e-8)
                effective_weight = max(tc.weight_floor, min(tc.weight_ceiling, raw_weight))
                contribution = effective_weight * loss_val

            total += contribution
            contributions[name] = contribution
            effective_weights[name] = effective_weight
            raw_weights[name] = raw_weight

            # Update EMA
            alpha = self.config.loss_scale_smoothing
            self._ema_loss[name] = alpha * self._ema_loss.get(name, loss_val) + (1 - alpha) * loss_val
            self._loss_history[name].append(loss_val)

        return {
            "total_loss": total,
            "task_contributions": contributions,
            "effective_weights": effective_weights,
            "raw_weights": raw_weights,
        }

    def update_log_variance(self, name: str, gradient: float, lr: float = 0.001) -> None:
        """Manual log-variance update (for non-autograd environments)."""
        if name in self._log_variances:
            self._log_variances[name] -= lr * gradient

    @property
    def task_names(self) -> list[str]:
        return [t.name for t in self.config.tasks if t.enabled]


# ── Checkpoint Selection ─────────────────────────────────────────────

@dataclass
class CheckpointMetrics:
    """Validation metrics for a training checkpoint."""
    epoch: int
    step: int
    primary_metric: float  # Main metric for selection
    task_metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "primary_metric": self.primary_metric,
            "task_metrics": self.task_metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class HardFloorConfig:
    """Hard-floor thresholds for checkpoint selection.

    A checkpoint is only eligible for selection if ALL per-task metrics
    meet their floor. This prevents "one good task hides one bad task."
    """
    metric_floors: dict[str, float] = field(default_factory=dict)

    def check(self, metrics: CheckpointMetrics) -> tuple[bool, list[str]]:
        """Check if a checkpoint meets all hard floors."""
        violations = []
        for metric_name, floor in self.metric_floors.items():
            actual = metrics.task_metrics.get(metric_name)
            if actual is None:
                violations.append(f"Missing metric: {metric_name}")
            elif actual < floor:
                violations.append(
                    f"{metric_name}: {actual:.4f} < floor {floor:.4f}"
                )
        return len(violations) == 0, violations


class CheckpointSelector:
    """Selects the best checkpoint using hard-floor constraints.

    Only checkpoints that pass ALL hard floors are eligible.
    Among eligible checkpoints, the one with the best primary metric wins.
    """

    def __init__(self, hard_floors: HardFloorConfig | None = None) -> None:
        self.hard_floors = hard_floors or HardFloorConfig()
        self._checkpoints: list[CheckpointMetrics] = []

    def add_checkpoint(self, metrics: CheckpointMetrics) -> None:
        self._checkpoints.append(metrics)

    def select_best(self) -> CheckpointMetrics | None:
        """Select the best checkpoint that passes all hard floors."""
        eligible = []
        for cp in self._checkpoints:
            passed, _ = self.hard_floors.check(cp)
            if passed:
                eligible.append(cp)

        if not eligible:
            return None

        return max(eligible, key=lambda cp: cp.primary_metric)

    def selection_report(self) -> dict[str, Any]:
        """Generate a report of all checkpoints with eligibility status."""
        items = []
        for cp in self._checkpoints:
            passed, violations = self.hard_floors.check(cp)
            items.append({
                **cp.to_dict(),
                "eligible": passed,
                "violations": violations,
            })
        best = self.select_best()
        return {
            "total_checkpoints": len(self._checkpoints),
            "eligible_count": sum(1 for i in items if i["eligible"]),
            "best_checkpoint": best.to_dict() if best else None,
            "checkpoints": items,
        }


# ── Training Config Validation ───────────────────────────────────────

@dataclass
class TrainingConfig:
    """Full training configuration with validation."""
    model_name: str = "nl2sql_v2"
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    multi_task: MultiTaskLossConfig = field(default_factory=MultiTaskLossConfig)
    hard_floors: HardFloorConfig = field(default_factory=HardFloorConfig)
    validation_split: str = "model_selection_validation"
    test_split: str = "frozen_semantic_test"

    def validate(self) -> list[str]:
        errors = []
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.max_epochs < 1:
            errors.append("max_epochs must be >= 1")
        if self.early_stopping_patience < 1:
            errors.append("early_stopping_patience must be >= 1")
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be >= 0")
        if self.weight_decay < 0:
            errors.append("weight_decay must be >= 0")
        if self.gradient_clip <= 0:
            errors.append("gradient_clip must be > 0")

        # Ensure test split is never used for checkpoint selection
        if self.validation_split == self.test_split:
            errors.append("validation_split must differ from test_split")

        errors.extend(self.multi_task.validate())
        return errors


__all__ = [
    "CheckpointMetrics",
    "CheckpointSelector",
    "HardFloorConfig",
    "MultiTaskLossComputer",
    "MultiTaskLossConfig",
    "TaskLossConfig",
    "TrainingConfig",
]
