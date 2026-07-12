"""Feature Flag Infrastructure — Gate 3 prerequisite.

Provides a central, typed feature flag system for controlling architecture
experiments independently. Every new architecture component (schema graph,
multi-head attention, capability head, etc.) ships behind a flag that defaults OFF.

Design:
- Flags are typed (bool, int, float, str) with defaults
- Runtime overrides via environment variables or config
- Audit log of flag evaluations for reproducibility
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FlagType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


@dataclass
class FlagDefinition:
    """A single feature flag definition."""
    name: str
    flag_type: FlagType
    default_value: Any
    description: str = ""
    gate: int = 0
    owner: str = ""

    def typed_default(self) -> Any:
        return _cast(self.default_value, self.flag_type)


@dataclass
class FlagEvaluation:
    """Record of a flag evaluation for audit trail."""
    flag_name: str
    resolved_value: Any
    source: str  # "default", "config", "env", "override"


class FeatureFlagRegistry:
    """Central registry for all feature flags.

    Resolution order (highest priority first):
    1. Runtime overrides (set_override)
    2. Environment variables (NL2SQL_FLAG_{NAME})
    3. Config file
    4. Default value
    """

    ENV_PREFIX = "NL2SQL_FLAG_"

    def __init__(self) -> None:
        self._definitions: dict[str, FlagDefinition] = {}
        self._overrides: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._evaluations: list[FlagEvaluation] = []

    def define(self, flag: FlagDefinition) -> None:
        self._definitions[flag.name] = flag

    def get(self, name: str) -> Any:
        """Resolve a flag value with full precedence chain."""
        defn = self._definitions.get(name)
        if defn is None:
            raise KeyError(f"Unknown feature flag: {name!r}")

        # 1. Runtime override
        if name in self._overrides:
            value = _cast(self._overrides[name], defn.flag_type)
            self._evaluations.append(FlagEvaluation(name, value, "override"))
            return value

        # 2. Environment variable
        env_key = f"{self.ENV_PREFIX}{name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            value = _cast(env_value, defn.flag_type)
            self._evaluations.append(FlagEvaluation(name, value, "env"))
            return value

        # 3. Config file
        if name in self._config:
            value = _cast(self._config[name], defn.flag_type)
            self._evaluations.append(FlagEvaluation(name, value, "config"))
            return value

        # 4. Default
        value = defn.typed_default()
        self._evaluations.append(FlagEvaluation(name, value, "default"))
        return value

    def get_bool(self, name: str) -> bool:
        return bool(self.get(name))

    def get_int(self, name: str) -> int:
        return int(self.get(name))

    def get_float(self, name: str) -> float:
        return float(self.get(name))

    def set_override(self, name: str, value: Any) -> None:
        if name not in self._definitions:
            raise KeyError(f"Unknown feature flag: {name!r}")
        self._overrides[name] = value

    def clear_override(self, name: str) -> None:
        self._overrides.pop(name, None)

    def clear_all_overrides(self) -> None:
        self._overrides.clear()

    def load_config(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._config.update(data.get("feature_flags", data))

    def load_config_dict(self, config: dict[str, Any]) -> None:
        self._config.update(config)

    @property
    def evaluation_log(self) -> list[FlagEvaluation]:
        return list(self._evaluations)

    def clear_evaluation_log(self) -> None:
        self._evaluations.clear()

    def all_flags(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "type": defn.flag_type.value,
                "default": defn.default_value,
                "current": self.get(name),
                "description": defn.description,
                "gate": defn.gate,
            }
            for name, defn in self._definitions.items()
        }

    @classmethod
    def default(cls) -> FeatureFlagRegistry:
        """Create registry with all Gate 3 architecture flags."""
        registry = cls()
        _register_defaults(registry)
        return registry


def _cast(value: Any, flag_type: FlagType) -> Any:
    if flag_type == FlagType.BOOL:
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        return bool(value)
    if flag_type == FlagType.INT:
        return int(value)
    if flag_type == FlagType.FLOAT:
        return float(value)
    return str(value)


def _register_defaults(registry: FeatureFlagRegistry) -> None:
    """Register all architecture flags, defaulting OFF."""
    flags = [
        FlagDefinition("enable_schema_graph", FlagType.BOOL, False,
                       "Use structured schema graph embeddings", gate=3),
        FlagDefinition("enable_multi_head_attention", FlagType.BOOL, False,
                       "Use multi-head cross-attention in encoder", gate=3),
        FlagDefinition("enable_capability_head", FlagType.BOOL, False,
                       "Separate capability prediction head", gate=3),
        FlagDefinition("enable_safety_head", FlagType.BOOL, False,
                       "Separate safety classification head", gate=3),
        FlagDefinition("enable_complexity_head", FlagType.BOOL, False,
                       "Query complexity prediction head", gate=3),
        FlagDefinition("enable_join_path_scorer", FlagType.BOOL, False,
                       "Bounded join-path enumeration scorer", gate=3),
        FlagDefinition("enable_hierarchical_decoder", FlagType.BOOL, False,
                       "Hierarchical scope-aware decoder", gate=3),
        FlagDefinition("enable_grammar_decoder", FlagType.BOOL, False,
                       "Grammar-constrained decoding", gate=3),
        FlagDefinition("enable_having", FlagType.BOOL, False,
                       "Allow HAVING clause in output", gate=1),
        FlagDefinition("enable_cte", FlagType.BOOL, False,
                       "Allow CTEs in output", gate=1),
        FlagDefinition("enable_subquery", FlagType.BOOL, False,
                       "Allow subqueries in output", gate=1),
        FlagDefinition("enable_window", FlagType.BOOL, False,
                       "Allow window functions in output", gate=1),
        FlagDefinition("enable_case", FlagType.BOOL, False,
                       "Allow CASE expressions in output", gate=1),
        FlagDefinition("enable_set_op", FlagType.BOOL, False,
                       "Allow UNION/INTERSECT/EXCEPT in output", gate=1),
        FlagDefinition("enable_exists", FlagType.BOOL, False,
                       "Allow EXISTS predicates in output", gate=1),
        FlagDefinition("enable_in_subquery", FlagType.BOOL, False,
                       "Allow IN (subquery) predicates in output", gate=1),
        FlagDefinition("max_join_path_depth", FlagType.INT, 4,
                       "Maximum join path enumeration depth", gate=3),
        FlagDefinition("schema_graph_edge_types", FlagType.INT, 8,
                       "Number of edge types in schema graph", gate=3),
        FlagDefinition("attention_heads", FlagType.INT, 4,
                       "Number of attention heads", gate=3),
        FlagDefinition("capability_loss_weight", FlagType.FLOAT, 0.1,
                       "Weight for capability head loss", gate=4),
        FlagDefinition("safety_loss_weight", FlagType.FLOAT, 0.1,
                       "Weight for safety head loss", gate=4),
    ]
    for flag in flags:
        registry.define(flag)


__all__ = [
    "FeatureFlagRegistry",
    "FlagDefinition",
    "FlagEvaluation",
    "FlagType",
]
