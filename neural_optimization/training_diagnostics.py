"""Training diagnostics tracker.

Records per-epoch metrics and produces JSON + Markdown reports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TrainingDiagnostics:
    """Accumulates per-epoch training diagnostics and writes reports."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.epochs: list[dict[str, Any]] = []
        self._start_time: float | None = None
        self.config_summary: dict[str, Any] = {}

    def set_config(self, config: dict[str, Any]) -> None:
        self.config_summary = {
            "optimizer": config.get("optimizer", {}).get("name", "unknown"),
            "activation": config.get("model", {}).get("activation", "unknown"),
            "learning_rate": config.get("optimizer", {}).get("learning_rate"),
            "batch_size": config.get("training", {}).get("batch_size"),
            "epochs": config.get("training", {}).get("epochs"),
            "gradient_clipping": config.get("training", {}).get("gradient_clipping"),
        }

    def start_training(self) -> None:
        self._start_time = time.time()

    def record_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float | None = None,
        epoch_time: float | None = None,
        loss_by_head: dict[str, float] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": train_metrics.get("loss", 0.0),
            "validation_total_loss": val_metrics.get("loss", 0.0),
            "intent_accuracy": val_metrics.get("intent_accuracy", 0.0),
            "base_table_accuracy": val_metrics.get("base_table_accuracy",
                                    val_metrics.get("metric_pointer_accuracy", 0.0)),
            "metric_column_accuracy": val_metrics.get("metric_column_accuracy", 0.0),
            "dimension_column_accuracy": val_metrics.get("dimension_column_accuracy", 0.0),
            "filter_column_accuracy": val_metrics.get("filter_column_accuracy", 0.0),
            "date_column_accuracy": val_metrics.get("date_column_accuracy", 0.0),
            "overall_slot_accuracy": val_metrics.get("overall_slot_accuracy", 0.0),
            "learning_rate": lr,
            "epoch_time_seconds": epoch_time,
        }
        if loss_by_head:
            row["loss_by_head"] = loss_by_head
        self.epochs.append(row)

    def best_epoch(self) -> dict[str, Any]:
        if not self.epochs:
            return {}
        return max(self.epochs, key=lambda e: e.get("overall_slot_accuracy", 0.0))

    def to_dict(self) -> dict[str, Any]:
        total_time = (time.time() - self._start_time) if self._start_time else None
        best = self.best_epoch()
        return {
            "config": self.config_summary,
            "total_training_time_seconds": total_time,
            "total_epochs": len(self.epochs),
            "best_epoch": best.get("epoch"),
            "best_overall_slot_accuracy": best.get("overall_slot_accuracy"),
            "epochs": self.epochs,
        }

    def save(self, output_dir: str | Path | None = None) -> None:
        target = Path(output_dir or self.output_dir or ".")
        target.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        (target / "training_diagnostics.json").write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8",
        )
        (target / "training_diagnostics.md").write_text(
            _render_markdown(data), encoding="utf-8",
        )


def _render_markdown(data: dict[str, Any]) -> str:
    lines = ["# Neural QueryIR Training Diagnostics", ""]
    cfg = data.get("config", {})
    lines.append("## Configuration")
    lines.append(f"- **Optimizer**: {cfg.get('optimizer', '—')}")
    lines.append(f"- **Activation**: {cfg.get('activation', '—')}")
    lines.append(f"- **Learning rate**: {cfg.get('learning_rate', '—')}")
    lines.append(f"- **Batch size**: {cfg.get('batch_size', '—')}")
    lines.append(f"- **Gradient clipping**: {cfg.get('gradient_clipping', '—')}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Total epochs**: {data.get('total_epochs', 0)}")
    lines.append(f"- **Best epoch**: {data.get('best_epoch', '—')}")
    lines.append(f"- **Best slot accuracy**: {data.get('best_overall_slot_accuracy', 0):.4f}")
    tt = data.get("total_training_time_seconds")
    if tt is not None:
        lines.append(f"- **Total training time**: {tt:.1f}s")
    lines.append("")
    epochs = data.get("epochs", [])
    if epochs:
        lines.append("## Per-Epoch Metrics")
        lines.append("")
        lines.append("| Epoch | Train Loss | Val Loss | Intent Acc | Slot Acc | LR | Time (s) |")
        lines.append("|------:|-----------:|---------:|-----------:|---------:|---:|---------:|")
        for e in epochs:
            lines.append(
                f"| {e.get('epoch', '—')} "
                f"| {e.get('train_total_loss', 0):.4f} "
                f"| {e.get('validation_total_loss', 0):.4f} "
                f"| {e.get('intent_accuracy', 0):.4f} "
                f"| {e.get('overall_slot_accuracy', 0):.4f} "
                f"| {e.get('learning_rate', '—')} "
                f"| {e.get('epoch_time_seconds', '—')} |"
            )
    lines.append("")
    best = data.get("best_epoch")
    if best:
        lines.append(f"## Recommended Next Action")
        lines.append(f"Best model from epoch {best}. Review validation metrics and run experiment grid for further tuning.")
    return "\n".join(lines)
