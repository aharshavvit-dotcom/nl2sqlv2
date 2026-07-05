"""Training configuration system for the Neural QueryIR Model.

Loads YAML config, supports CLI overrides, and saves the effective config
that was actually used for a training run.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ── Defaults ──────────────────────────────────────────────────────────────

_DEFAULT_MODEL = {
    "model_version": "schema_aware_queryir_v1",
    "architecture": "schema_aware_queryir",
    "encoder": "bigru",
    "hidden_dim": 192,
    "embedding_dim": 128,
    "candidate_hidden_dim": 128,
    "feed_forward_dim": 256,
    "activation": "gelu",
    "dropout": 0.25,
    "pointer_dropout": 0.30,
    "layer_norm": True,
    "feed_forward_heads": True,
    "max_question_len": 64,
    "max_schema_len": 320,
    "max_candidate_tokens": 16,
    "max_tables": 64,
    "max_columns": 256,
    "relation_aware_attention": {
        "enabled": False,
        "relation_bias_mode": "schema_pairwise_relation_bias",
        "pairwise_relation_matrix": True,
        "relation_types": [
            "same_table",
            "table_has_column",
            "column_belongs_to_table",
            "fk_to_pk",
            "pk_to_fk",
            "primary_key",
            "foreign_key_column",
            "same_column_name",
            "same_data_type",
            "unrelated",
        ],
        "bias_init": 0.0,
    },
}

_DEFAULT_OPTIMIZER = {
    "name": "adamw",
    "learning_rate": 0.0007,
    "weight_decay": 0.0001,
    "pointer_head_weight_decay": 0.001,
    "momentum": 0.9,
    "nesterov": False,
}

_DEFAULT_SCHEDULER = {
    "name": "reduce_on_plateau",
    "factor": 0.5,
    "patience": 2,
    "min_lr": 0.000001,
    "step_size": 5,
    "t_max": 10,
}

_DEFAULT_TRAINING = {
    "batch_size": 8,
    "epochs": 10,
    "gradient_clipping": 1.0,
    "early_stopping_patience": 2,
    "seed": 42,
    "save_best_metric": "loss",
    "save_best_mode": "min",
}

_DEFAULT_LOSS = {
    "intent": 1.0,
    "base_table": 1.2,
    "metric_column": 1.0,
    "metric_aggregation": 0.8,
    "metric_expression_type": 0.8,
    "dimension_column": 1.0,
    "filter_column": 1.2,
    "date_column": 1.0,
    "date_grain": 0.6,
    "date_filter_type": 0.6,
    "filter_operator": 0.8,
    "aggregation": 0.8,
    "order_direction": 0.6,
    "limit_bucket": 0.4,
    "hard_negative": 0.3,
}

_DEFAULT_DATA = {
    "train_path": "data/processed/generic_ir_train.jsonl",
    "validation_path": "data/processed/generic_ir_validation.jsonl",
    "hard_negatives_path": "data/processed/generic_ir_hard_negatives.jsonl",
    "max_examples": 0,
}

_DEFAULT_OUTPUT = {
    "output_dir": "artifacts/neural_ir_model",
    "save_diagnostics": True,
    "save_effective_config": True,
}


# ── Dataclass ─────────────────────────────────────────────────────────────

@dataclass
class NeuralTrainingConfig:
    """Complete configuration for optimized neural QueryIR training."""

    model: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_MODEL))
    optimizer: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_OPTIMIZER))
    scheduler: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_SCHEDULER))
    training: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_TRAINING))
    loss: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_LOSS))
    data: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_DATA))
    output: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_OUTPUT))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Loaders / helpers ─────────────────────────────────────────────────────

def load_training_config(path: str | Path) -> NeuralTrainingConfig:
    """Load a ``NeuralTrainingConfig`` from a YAML file.

    Any section not present in the file receives default values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_config(raw)


def merge_cli_overrides(config: NeuralTrainingConfig, overrides: dict[str, Any]) -> NeuralTrainingConfig:
    """Apply flat CLI overrides on top of an existing config.

    Supported keys (mapped to their sections):
      ``epochs``, ``batch_size`` → ``training``
      ``optimizer``, ``learning_rate`` → ``optimizer``
      ``activation`` → ``model``
      ``output_dir`` → ``output``
      ``seed`` → ``training``
      ``max_examples`` → ``data``
      ``gradient_clipping`` → ``training``
    """
    _map = {
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "optimizer": ("optimizer", "name"),
        "learning_rate": ("optimizer", "learning_rate"),
        "activation": ("model", "activation"),
        "output_dir": ("output", "output_dir"),
        "seed": ("training", "seed"),
        "max_examples": ("data", "max_examples"),
        "gradient_clipping": ("training", "gradient_clipping"),
        "train": ("data", "train_path"),
        "validation": ("data", "validation_path"),
        "hard_negatives": ("data", "hard_negatives_path"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key in _map:
            section, field_name = _map[key]
            getattr(config, section)[field_name] = value
    return config


def save_effective_config(config: NeuralTrainingConfig, output_path: str | Path) -> None:
    """Write the config that was actually used to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False), encoding="utf-8")


# ── Private helpers ───────────────────────────────────────────────────────

def _build_config(raw: dict[str, Any]) -> NeuralTrainingConfig:
    """Merge raw dict from YAML into a fully-defaulted config."""
    return NeuralTrainingConfig(
        model={**_DEFAULT_MODEL, **(raw.get("model") or {})},
        optimizer={**_DEFAULT_OPTIMIZER, **(raw.get("optimizer") or {})},
        scheduler={**_DEFAULT_SCHEDULER, **(raw.get("scheduler") or {})},
        training={**_DEFAULT_TRAINING, **(raw.get("training") or {})},
        loss={**_DEFAULT_LOSS, **(raw.get("loss") or {})},
        data={**_DEFAULT_DATA, **(raw.get("data") or {})},
        output={**_DEFAULT_OUTPUT, **(raw.get("output") or {})},
    )
