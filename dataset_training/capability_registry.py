"""Active Capability Registry — tracks which capabilities are available,
which are behind feature flags, and which have sufficient training coverage.

Gate 2: Data Readiness — provides the single source of truth for capability
maturity throughout the system (training, inference, evaluation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CapabilityStatus(str, Enum):
    """Lifecycle status of a capability."""
    ACTIVE = "active"           # Production-ready, fully trained
    FLAGGED = "flagged"         # Behind feature flag, in training
    MASKED = "masked"           # Known but intentionally disabled
    EXPERIMENTAL = "experimental"  # In development, not yet validated
    DEPRECATED = "deprecated"   # Scheduled for removal


@dataclass
class CapabilityEntry:
    """A single capability in the registry."""
    name: str
    status: CapabilityStatus
    description: str = ""
    feature_flag: str | None = None
    min_training_examples: int = 50
    current_training_count: int = 0
    validation_accuracy: float | None = None
    owner: str = ""
    gate: int = 0  # Gate at which this capability was introduced

    @property
    def training_sufficient(self) -> bool:
        return self.current_training_count >= self.min_training_examples

    @property
    def production_ready(self) -> bool:
        return (
            self.status == CapabilityStatus.ACTIVE
            and self.training_sufficient
            and (self.validation_accuracy is None or self.validation_accuracy >= 0.8)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "feature_flag": self.feature_flag,
            "min_training_examples": self.min_training_examples,
            "current_training_count": self.current_training_count,
            "validation_accuracy": self.validation_accuracy,
            "owner": self.owner,
            "gate": self.gate,
            "training_sufficient": self.training_sufficient,
            "production_ready": self.production_ready,
        }


class ActiveCapabilityRegistry:
    """Centralized registry of all system capabilities and their maturity state.

    Usage:
        registry = ActiveCapabilityRegistry.default()
        registry.is_enabled("OR_FILTER")  # True if active or flagged-on
        registry.get_training_gap()  # Capabilities needing more data
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityEntry] = {}
        self._feature_flags: dict[str, bool] = {}

    def register(self, entry: CapabilityEntry) -> None:
        self._capabilities[entry.name] = entry

    def get(self, name: str) -> CapabilityEntry | None:
        return self._capabilities.get(name)

    def is_enabled(self, name: str) -> bool:
        """Check if a capability is available for inference."""
        entry = self._capabilities.get(name)
        if entry is None:
            return False
        if entry.status == CapabilityStatus.ACTIVE:
            return True
        if entry.status == CapabilityStatus.FLAGGED and entry.feature_flag:
            return self._feature_flags.get(entry.feature_flag, False)
        return False

    def set_feature_flag(self, flag: str, enabled: bool) -> None:
        self._feature_flags[flag] = enabled

    def update_training_count(self, name: str, count: int) -> None:
        entry = self._capabilities.get(name)
        if entry is not None:
            entry.current_training_count = count

    def update_validation_accuracy(self, name: str, accuracy: float) -> None:
        entry = self._capabilities.get(name)
        if entry is not None:
            entry.validation_accuracy = accuracy

    def get_training_gap(self) -> list[CapabilityEntry]:
        """Return capabilities that need more training data."""
        return [
            entry for entry in self._capabilities.values()
            if entry.status in {CapabilityStatus.ACTIVE, CapabilityStatus.FLAGGED}
            and not entry.training_sufficient
        ]

    def get_production_ready(self) -> list[CapabilityEntry]:
        return [e for e in self._capabilities.values() if e.production_ready]

    def get_flagged(self) -> list[CapabilityEntry]:
        return [e for e in self._capabilities.values() if e.status == CapabilityStatus.FLAGGED]

    def coverage_report(self) -> dict[str, Any]:
        """Generate a comprehensive coverage report."""
        entries = list(self._capabilities.values())
        by_status = {}
        for entry in entries:
            status = entry.status.value
            by_status.setdefault(status, []).append(entry.name)

        production_ready = [e for e in entries if e.production_ready]
        training_gap = self.get_training_gap()

        return {
            "total_capabilities": len(entries),
            "production_ready": len(production_ready),
            "training_gap": len(training_gap),
            "by_status": {k: len(v) for k, v in by_status.items()},
            "gap_details": [
                {"name": e.name, "current": e.current_training_count, "needed": e.min_training_examples}
                for e in training_gap
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": {name: entry.to_dict() for name, entry in self._capabilities.items()},
            "feature_flags": dict(self._feature_flags),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ActiveCapabilityRegistry:
        data = json.loads(Path(path).read_text())
        registry = cls()
        for name, entry_data in data.get("capabilities", {}).items():
            registry.register(CapabilityEntry(
                name=entry_data["name"],
                status=CapabilityStatus(entry_data["status"]),
                description=entry_data.get("description", ""),
                feature_flag=entry_data.get("feature_flag"),
                min_training_examples=entry_data.get("min_training_examples", 50),
                current_training_count=entry_data.get("current_training_count", 0),
                validation_accuracy=entry_data.get("validation_accuracy"),
                owner=entry_data.get("owner", ""),
                gate=entry_data.get("gate", 0),
            ))
        registry._feature_flags = dict(data.get("feature_flags", {}))
        return registry

    @classmethod
    def default(cls) -> ActiveCapabilityRegistry:
        """Create the default registry with all known capabilities."""
        registry = cls()
        _register_defaults(registry)
        return registry


def _register_defaults(registry: ActiveCapabilityRegistry) -> None:
    """Register all known capabilities with their default status."""
    # Gate 0 (Phase 2A/2B) — Active
    for name in [
        "SIMPLE_SELECT", "WHERE_FILTER", "AND_FILTER", "COMPARISON",
        "IN_FILTER", "NULL_CHECK", "BETWEEN", "LIKE_FILTER",
        "JOIN", "GROUP_BY", "ORDER_BY", "AGGREGATION", "LIMIT",
    ]:
        registry.register(CapabilityEntry(
            name=name, status=CapabilityStatus.ACTIVE,
            description=f"Core capability: {name}", gate=0,
        ))

    # Gate 1 — Flagged (new v2 constructs)
    for name, flag in [
        ("OR_FILTER", "enable_or_rendering"),
        ("HAVING", "enable_having"),
        ("SUBQUERY", "enable_subquery"),
        ("CTE", "enable_cte"),
        ("WINDOW_FUNCTION", "enable_window"),
        ("CASE_EXPRESSION", "enable_case"),
        ("SET_OPERATION", "enable_set_op"),
        ("EXISTS_PREDICATE", "enable_exists"),
        ("IN_SUBQUERY", "enable_in_subquery"),
    ]:
        registry.register(CapabilityEntry(
            name=name, status=CapabilityStatus.FLAGGED,
            feature_flag=flag,
            description=f"Gate 1 capability: {name}",
            gate=1,
        ))

    # Gate 3 — Experimental (architecture)
    for name in [
        "SCHEMA_GRAPH_EMBEDDING", "MULTI_HEAD_ATTENTION",
        "CAPABILITY_HEAD", "SAFETY_HEAD", "COMPLEXITY_HEAD",
        "JOIN_PATH_SCORER", "HIERARCHICAL_DECODER",
    ]:
        registry.register(CapabilityEntry(
            name=name, status=CapabilityStatus.EXPERIMENTAL,
            description=f"Gate 3 architecture: {name}", gate=3,
        ))


__all__ = [
    "ActiveCapabilityRegistry",
    "CapabilityEntry",
    "CapabilityStatus",
]
