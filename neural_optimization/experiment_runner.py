"""Experiment runner for neural training hyperparameter search.

Iterates over a grid of configs, trains each, and collects results.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .training_config import NeuralTrainingConfig, load_training_config, save_effective_config


class ExperimentRunner:
    """Runs a grid of training experiments.

    Parameters
    ----------
    base_config:
        The base ``NeuralTrainingConfig`` that experiments override.
    grid:
        List of experiment dicts, each with ``name`` and overrides.
    output_dir:
        Root directory for experiment outputs.
    """

    def __init__(
        self,
        base_config: NeuralTrainingConfig,
        grid: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> None:
        self.base_config = base_config
        self.grid = grid
        self.output_dir = Path(output_dir)
        self.results: list[dict[str, Any]] = []

    def run(self, train_fn: Any) -> list[dict[str, Any]]:
        """Execute all experiments in the grid.

        Parameters
        ----------
        train_fn:
            A callable ``(config: NeuralTrainingConfig, output_dir: Path) -> dict``
            that runs a single training session and returns metrics.

        Returns
        -------
        List of per-experiment result dicts.
        """
        self.results = []
        for i, spec in enumerate(self.grid, 1):
            name = spec.get("name", f"experiment_{i}")
            print(f"\n{'='*60}")
            print(f"Experiment {i}/{len(self.grid)}: {name}")
            print(f"{'='*60}")
            config = self._merge_experiment(spec)
            exp_dir = self.output_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)
            save_effective_config(config, exp_dir / "effective_config.yaml")
            start = time.time()
            try:
                metrics = train_fn(config, exp_dir)
            except Exception as exc:
                print(f"  Experiment '{name}' failed: {exc}")
                metrics = {"error": str(exc)}
            elapsed = time.time() - start
            result = {
                "name": name,
                "optimizer": config.optimizer.get("name"),
                "activation": config.model.get("activation"),
                "learning_rate": config.optimizer.get("learning_rate"),
                "training_time_seconds": elapsed,
                "metrics": metrics,
            }
            self.results.append(result)
            print(f"  Completed in {elapsed:.1f}s")
        return self.results

    def _merge_experiment(self, spec: dict[str, Any]) -> NeuralTrainingConfig:
        cfg = NeuralTrainingConfig(
            model=copy.deepcopy(self.base_config.model),
            optimizer=copy.deepcopy(self.base_config.optimizer),
            scheduler=copy.deepcopy(self.base_config.scheduler),
            training=copy.deepcopy(self.base_config.training),
            loss=copy.deepcopy(self.base_config.loss),
            data=copy.deepcopy(self.base_config.data),
            output=copy.deepcopy(self.base_config.output),
        )
        for section in ("model", "optimizer", "scheduler", "training", "loss", "data", "output"):
            overrides = spec.get(section, {})
            if overrides:
                getattr(cfg, section).update(overrides)
        return cfg


def load_experiment_grid(path: str | Path) -> list[dict[str, Any]]:
    """Load an experiment grid from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return raw.get("experiments", [])
