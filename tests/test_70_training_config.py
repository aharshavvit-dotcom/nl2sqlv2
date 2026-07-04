"""Tests for neural_optimization.training_config."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from neural_optimization.training_config import (
    NeuralTrainingConfig,
    load_training_config,
    merge_cli_overrides,
    save_effective_config,
)


class TestNeuralTrainingConfig:
    def test_defaults(self):
        cfg = NeuralTrainingConfig()
        assert cfg.model["activation"] == "gelu"
        assert cfg.optimizer["name"] == "adamw"
        assert cfg.training["batch_size"] == 8
        assert cfg.loss["base_table"] == 1.2
        assert cfg.scheduler["name"] == "reduce_on_plateau"

    def test_to_dict(self):
        cfg = NeuralTrainingConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "optimizer" in d
        assert "loss" in d

    def test_load_from_yaml(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump({
            "model": {"activation": "relu", "dropout": 0.1},
            "optimizer": {"name": "adam"},
            "training": {"epochs": 5},
        }), encoding="utf-8")
        cfg = load_training_config(p)
        assert cfg.model["activation"] == "relu"
        assert cfg.model["dropout"] == 0.1
        assert cfg.optimizer["name"] == "adam"
        assert cfg.training["epochs"] == 5
        # Defaults preserved for missing keys
        assert cfg.model["hidden_dim"] == 192
        assert cfg.scheduler["name"] == "reduce_on_plateau"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_training_config("does_not_exist.yaml")

    def test_cli_overrides(self):
        cfg = NeuralTrainingConfig()
        cfg = merge_cli_overrides(cfg, {
            "epochs": 20,
            "batch_size": 16,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "activation": "relu",
        })
        assert cfg.training["epochs"] == 20
        assert cfg.training["batch_size"] == 16
        assert cfg.optimizer["name"] == "sgd"
        assert cfg.optimizer["learning_rate"] == 0.01
        assert cfg.model["activation"] == "relu"

    def test_cli_overrides_none_ignored(self):
        cfg = NeuralTrainingConfig()
        original_lr = cfg.optimizer["learning_rate"]
        cfg = merge_cli_overrides(cfg, {"learning_rate": None})
        assert cfg.optimizer["learning_rate"] == original_lr

    def test_save_effective_config(self, tmp_path):
        cfg = NeuralTrainingConfig()
        out = tmp_path / "effective.yaml"
        save_effective_config(cfg, out)
        assert out.exists()
        loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert loaded["model"]["activation"] == "gelu"
        assert loaded["optimizer"]["name"] == "adamw"


def test_integrated_training_modes_are_explicit_and_safe():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "debug_training.yaml": "debug",
        "baseline_training.yaml": "baseline",
        "training.yaml": "production",
    }
    for name, mode in expected.items():
        payload = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
        assert payload["quality_gate"]["mode"] == mode
        if mode != "production":
            assert payload["bundle"]["promote_if_quality_gate_passes"] is False
            assert payload["pipeline"]["promote_if_passed"] is False
