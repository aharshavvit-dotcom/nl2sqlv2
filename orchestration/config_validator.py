"""Configuration validator for production training runs.

Validates that effective neural training configs match canonical requirements
for the given pipeline mode (production, baseline, debug, smoke).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Canonical production/baseline neural training settings.
# Debug/smoke configs may override ONLY when explicitly marked.
CANONICAL_NEURAL_SETTINGS = {
    "epochs": 10,
    "batch_size": 8,
    "early_stopping_patience": 2,
    "save_best_metric": "loss",
    "save_best_mode": "min",
    "weight_decay": 0.0001,
    "pointer_head_weight_decay": 0.001,
    "pointer_dropout": 0.30,
}


class ConfigValidationError(ValueError):
    """Raised when a training config fails canonical validation."""


def validate_neural_config(
    effective_config: dict[str, Any],
    mode: str,
    *,
    allow_override: bool = False,
) -> list[str]:
    """Validate effective neural config against canonical requirements.

    Args:
        effective_config: The resolved neural training configuration.
        mode: Pipeline mode — 'production', 'baseline', 'debug', or 'smoke'.
        allow_override: If True, deviations are warnings not errors.

    Returns:
        List of issues found (empty if valid).
    """
    issues: list[str] = []
    mode = str(mode or "production").lower()

    if mode in {"debug", "smoke"}:
        # Debug/smoke configs may override canonical settings
        if not allow_override:
            issues.append(
                f"config_mode={mode} overrides canonical settings without "
                f"allow_neural_config_override=true"
            )
        return issues

    # Production and baseline modes must match canonical settings exactly
    for key, expected in CANONICAL_NEURAL_SETTINGS.items():
        actual = effective_config.get(key)
        if actual is None:
            issues.append(f"missing_canonical_setting: {key} (expected {expected})")
            continue
        # Compare with type coercion for floats
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-9:
                issues.append(
                    f"canonical_mismatch: {key}={actual} (expected {expected})"
                )
        elif isinstance(expected, int):
            if int(actual) != expected:
                issues.append(
                    f"canonical_mismatch: {key}={actual} (expected {expected})"
                )
        elif str(actual) != str(expected):
            issues.append(
                f"canonical_mismatch: {key}={actual} (expected {expected})"
            )

    return issues


def validate_pipeline_config(
    config: dict[str, Any],
    *,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Validate a full training pipeline config.

    Returns:
        Dict with 'valid', 'issues', 'warnings', and 'effective_settings'.
    """
    mode = (config.get("quality_gate") or {}).get("mode", "baseline")
    profile = config.get("profile", mode)
    allow_override = bool(config.get("allow_neural_config_override", False))

    effective_neural = config.get("_effective_neural_config") or {}
    issues = validate_neural_config(
        effective_neural,
        mode=profile,
        allow_override=allow_override,
    )

    # Validate promotion settings are consistent with mode
    warnings: list[str] = []
    pipeline = config.get("pipeline") or {}
    bundle = config.get("bundle") or {}

    if mode in {"debug", "smoke"}:
        if pipeline.get("promote_if_passed", False):
            issues.append("debug/smoke pipeline must not set promote_if_passed=true")
        if bundle.get("promote_if_quality_gate_passes", False):
            issues.append("debug/smoke pipeline must not set promote_if_quality_gate_passes=true")

    if mode == "baseline":
        if pipeline.get("promote_if_passed", False):
            warnings.append("baseline pipeline has promote_if_passed=true; promotion will be blocked by quality gate mode")

    result = {
        "valid": len(issues) == 0,
        "mode": mode,
        "profile": profile,
        "issues": issues,
        "warnings": warnings,
        "effective_settings": {
            key: effective_neural.get(key) for key in CANONICAL_NEURAL_SETTINGS
        },
    }

    if raise_on_failure and issues:
        raise ConfigValidationError(
            f"Config validation failed for mode={mode}: " + "; ".join(issues)
        )

    return result
