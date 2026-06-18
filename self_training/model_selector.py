"""Model Selector — selects the best model across self-training iterations.

Compares models by validation metrics, runs head-to-head comparisons, and
promotes the best model to the production artifact directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class ModelSelector:
    """Selects and promotes the best model from self-training iterations."""

    def select_best(
        self,
        iterations: list[dict[str, Any]],
        metric: str = "overall_slot_accuracy",
    ) -> int:
        """Return the iteration index with the best *metric* value.

        Parameters
        ----------
        iterations:
            List of per-iteration metric dicts (as from ImprovementTracker).
        metric:
            The metric name to maximise.

        Returns
        -------
        Index of the best iteration (0-based).
        """

        if not iterations:
            return 0

        best_idx = 0
        best_val = -1.0
        for idx, entry in enumerate(iterations):
            val = float(entry.get(metric, 0.0))
            if val > best_val:
                best_val = val
                best_idx = idx
        return best_idx

    def compare_models(
        self,
        model_a_metrics: dict[str, Any],
        model_b_metrics: dict[str, Any],
        primary_metric: str = "overall_slot_accuracy",
    ) -> dict[str, Any]:
        """Head-to-head comparison of two models by their metric dicts.

        Returns a dict with ``winner``, ``metric_diff``, and per-metric
        comparison.
        """

        a_val = float(model_a_metrics.get(primary_metric, 0.0))
        b_val = float(model_b_metrics.get(primary_metric, 0.0))

        per_metric: dict[str, dict[str, Any]] = {}
        all_keys = sorted(set(model_a_metrics) | set(model_b_metrics))
        for k in all_keys:
            try:
                va = float(model_a_metrics.get(k, 0.0))
                vb = float(model_b_metrics.get(k, 0.0))
            except (TypeError, ValueError):
                continue
            per_metric[k] = {
                "model_a": round(va, 6),
                "model_b": round(vb, 6),
                "diff": round(vb - va, 6),
                "winner": "model_b" if vb > va else ("model_a" if va > vb else "tie"),
            }

        return {
            "primary_metric": primary_metric,
            "model_a_score": round(a_val, 6),
            "model_b_score": round(b_val, 6),
            "metric_diff": round(b_val - a_val, 6),
            "winner": "model_b" if b_val > a_val else ("model_a" if a_val > b_val else "tie"),
            "per_metric": per_metric,
        }

    def promote_best(
        self,
        best_dir: str | Path,
        production_dir: str | Path,
    ) -> dict[str, Any]:
        """Copy the best model artifacts to the production directory.

        Returns a summary dict describing what was copied.
        """

        best = Path(best_dir)
        prod = Path(production_dir)

        if not best.exists():
            return {"success": False, "error": f"Source directory does not exist: {best}"}

        prod.mkdir(parents=True, exist_ok=True)

        copied_files: list[str] = []
        for item in best.iterdir():
            if item.is_file():
                dest = prod / item.name
                shutil.copy2(item, dest)
                copied_files.append(item.name)

        # Write promotion metadata
        promotion_meta = {
            "source_dir": str(best),
            "production_dir": str(prod),
            "files_copied": copied_files,
        }
        (prod / "promotion_metadata.json").write_text(
            json.dumps(promotion_meta, indent=2),
            encoding="utf-8",
        )

        return {"success": True, "files_copied": copied_files, **promotion_meta}
