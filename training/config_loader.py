"""Resolve and validate the neural configuration used by integrated training.

This module exists because pipeline YAML and neural YAML are separate inputs;
keeping their merge policy here prevents each training entry point from
inventing a different effective configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NEURAL_CONFIG = Path("configs/neural_training_default.yaml")
_REQUIRED_CANONICAL = {
    "epochs": 10,
    "batch_size": 8,
    "save_best_metric": "loss",
    "save_best_mode": "min",
    "early_stopping_patience": 2,
    "weight_decay": 0.0001,
    "pointer_head_weight_decay": 0.001,
    "pointer_dropout": 0.30,
}


def resolve_effective_neural_config(
    pipeline_config: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Return the audited neural settings and reject unsafe config drift."""
    neural = pipeline_config.get("neural") or {}
    mode = str((pipeline_config.get("quality_gate") or {}).get("mode") or "baseline").lower()
    profile = str(pipeline_config.get("profile") or mode).lower()
    configured = Path(str(neural.get("config") or CANONICAL_NEURAL_CONFIG))
    config_path = configured if configured.is_absolute() else root / configured
    if not config_path.exists():
        raise ValueError(f"Neural config does not exist: {configured.as_posix()}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training = dict(raw.get("training") or {})
    optimizer = dict(raw.get("optimizer") or {})
    model = dict(raw.get("model") or {})
    effective: dict[str, Any] = {
        "config_path": _relative_path(config_path, root),
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "save_best_metric": training.get("save_best_metric"),
        "save_best_mode": training.get("save_best_mode"),
        "early_stopping_patience": training.get("early_stopping_patience"),
        "weight_decay": optimizer.get("weight_decay"),
        "pointer_head_weight_decay": optimizer.get("pointer_head_weight_decay"),
        "dropout": model.get("dropout"),
        "pointer_dropout": model.get("pointer_dropout"),
        "override_sources": [],
        "debug_override_used": False,
        "not_production_training": mode in {"debug", "smoke"},
    }

    pipeline_overrides = {
        key: neural[key]
        for key in ("epochs", "batch_size")
        if key in neural and neural[key] is not None
    }
    override_allowed = bool(pipeline_config.get("allow_neural_config_override", False))
    dev_profile = mode in {"debug", "smoke"} and profile in {"debug", "smoke"}
    if pipeline_overrides and not (override_allowed and dev_profile):
        conflicts = {
            key: {"pipeline": value, "neural_config": effective.get(key)}
            for key, value in pipeline_overrides.items()
            if value != effective.get(key)
        }
        if conflicts:
            raise ValueError(
                "Pipeline neural overrides conflict with the canonical neural config: "
                + json.dumps(conflicts, sort_keys=True)
            )
        # Equal duplicate values are harmless, but the neural file remains the source of truth.
    elif pipeline_overrides:
        effective.update(pipeline_overrides)
        effective["override_sources"] = [f"neural.{key}" for key in sorted(pipeline_overrides)]
        effective["debug_override_used"] = True

    canonical_path = (root / CANONICAL_NEURAL_CONFIG).resolve()
    is_canonical_path = config_path.resolve() == canonical_path
    if mode in {"production", "release", "baseline"}:
        if not is_canonical_path:
            raise ValueError(
                f"{mode} training must use {CANONICAL_NEURAL_CONFIG.as_posix()}, "
                f"not {effective['config_path']}"
            )
        mismatches = {
            key: {"actual": effective.get(key), "required": required}
            for key, required in _REQUIRED_CANONICAL.items()
            if effective.get(key) != required
        }
        if mismatches:
            raise ValueError(
                f"Invalid canonical neural settings for {mode} training: "
                + json.dumps(mismatches, sort_keys=True)
            )
    elif not is_canonical_path and not (override_allowed and dev_profile):
        raise ValueError(
            "A non-canonical neural config is allowed only for an explicit debug/smoke profile "
            "with allow_neural_config_override=true."
        )

    hash_payload = {key: effective.get(key) for key in sorted(_REQUIRED_CANONICAL)}
    effective["effective_config_hash"] = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return effective


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
