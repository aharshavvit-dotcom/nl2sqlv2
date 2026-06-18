"""Experiment reporter — generates comparison reports from experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentReporter:
    """Generates summary JSON and Markdown from experiment results."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def best_experiment(self, metric: str = "overall_slot_accuracy") -> dict[str, Any] | None:
        """Return the experiment with the highest *metric*."""
        valid = [r for r in self.results if "error" not in (r.get("metrics") or {})]
        if not valid:
            return None
        return max(valid, key=lambda r: float((r.get("metrics") or {}).get(metric, 0)))

    def summary(self) -> dict[str, Any]:
        best = self.best_experiment()
        return {
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if "error" not in (r.get("metrics") or {})),
            "failed": sum(1 for r in self.results if "error" in (r.get("metrics") or {})),
            "best_experiment": best.get("name") if best else None,
            "best_metric": float((best.get("metrics") or {}).get("overall_slot_accuracy", 0)) if best else None,
            "experiments": self.results,
        }

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        (out / "experiment_summary.json").write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8",
        )
        (out / "experiment_summary.md").write_text(
            self._render_markdown(data), encoding="utf-8",
        )

    @staticmethod
    def _render_markdown(data: dict[str, Any]) -> str:
        lines = ["# Neural Training Experiment Summary", ""]
        lines.append(f"- **Total experiments**: {data.get('total_experiments', 0)}")
        lines.append(f"- **Successful**: {data.get('successful', 0)}")
        lines.append(f"- **Failed**: {data.get('failed', 0)}")
        lines.append(f"- **Best experiment**: {data.get('best_experiment', '—')}")
        lines.append(f"- **Best slot accuracy**: {data.get('best_metric', 0):.4f}" if data.get("best_metric") else "")
        lines.append("")
        lines.append("## Results")
        lines.append("")
        lines.append("| Name | Optimizer | Activation | LR | Slot Acc | Time (s) | Status |")
        lines.append("|------|-----------|------------|---:|--------:|---------:|--------|")
        for r in data.get("experiments", []):
            metrics = r.get("metrics") or {}
            status = "✗ " + str(metrics.get("error", ""))[:30] if "error" in metrics else "✓"
            sa = f"{float(metrics.get('overall_slot_accuracy', 0)):.4f}" if "error" not in metrics else "—"
            lines.append(
                f"| {r.get('name', '—')} "
                f"| {r.get('optimizer', '—')} "
                f"| {r.get('activation', '—')} "
                f"| {r.get('learning_rate', '—')} "
                f"| {sa} "
                f"| {r.get('training_time_seconds', 0):.1f} "
                f"| {status} |"
            )
        return "\n".join(lines)
