"""Checkpoint manager for neural model training.

Saves best and last model checkpoints along with metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Manages best/last model checkpoint persistence.

    Parameters
    ----------
    output_dir:
        Directory where checkpoints are saved.
    metric_name:
        Name of the metric used to decide "best" (e.g. ``validation_gold_score``).
    mode:
        ``"max"`` or ``"min"`` — whether higher or lower metric is better.
    """

    def __init__(
        self,
        output_dir: str | Path,
        metric_name: str = "validation_gold_score",
        mode: str = "max",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self._best_value: float | None = None
        self._best_epoch: int | None = None

    def maybe_save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> bool:
        """Save checkpoint if *metrics[metric_name]* is the new best.

        Returns ``True`` if a new best was saved.
        """
        value = float(metrics.get(self.metric_name,
                       metrics.get("overall_slot_accuracy",
                       metrics.get("sql_validation_rate", 0.0))))
        is_better = self._is_improvement(value)
        if is_better:
            self._best_value = value
            self._best_epoch = epoch
            self._save(
                model, optimizer, epoch, metrics, config,
                self.output_dir / "best_model.pt",
            )
            self._write_metadata(epoch, metrics, config, is_best=True)
        return is_better

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Save the most recent checkpoint regardless of metric."""
        self._save(
            model, optimizer, epoch, metrics, config,
            self.output_dir / "last_model.pt",
        )

    def load_best(self) -> dict[str, Any] | None:
        """Load the best checkpoint dict, or ``None`` if it doesn't exist."""
        path = self.output_dir / "best_model.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    # ── private ───────────────────────────────────────────────────────

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.mode == "max":
            return value > self._best_value
        return value < self._best_value

    @staticmethod
    def _save(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
        config: dict[str, Any] | None,
        path: Path,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config,
            },
            path,
        )

    def _write_metadata(
        self,
        epoch: int,
        metrics: dict[str, Any],
        config: dict[str, Any] | None,
        is_best: bool,
    ) -> None:
        meta = {
            "best_epoch": epoch,
            "best_metric_name": self.metric_name,
            "best_metric_value": self._best_value,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
        }
        if config:
            meta["optimizer"] = config.get("optimizer", {}).get("name")
            meta["activation"] = config.get("model", {}).get("activation")
        (self.output_dir / "checkpoint_metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8",
        )
