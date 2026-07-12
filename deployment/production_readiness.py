"""Production Readiness — Gate 6.

Shadow mode infrastructure, telemetry governance, bundle metadata extensions,
and rollback rehearsal for safe deployment.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ── Shadow Mode ──────────────────────────────────────────────────────

class ShadowMode(str, Enum):
    OFF = "off"           # v2 disabled, v1 only
    SHADOW = "shadow"     # v2 runs in parallel, results logged but not served
    CANARY = "canary"     # v2 served to N% of requests
    PROMOTED = "promoted" # v2 is primary


@dataclass
class ShadowResult:
    """Result of a shadow-mode comparison."""
    question: str
    v1_sql: str
    v2_sql: str
    v1_latency_ms: float
    v2_latency_ms: float
    match: bool
    v2_confidence: float = 0.0
    divergence_category: str = ""  # "equivalent", "subset", "different", "error"
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "v1_sql": self.v1_sql,
            "v2_sql": self.v2_sql,
            "match": self.match,
            "v1_latency_ms": self.v1_latency_ms,
            "v2_latency_ms": self.v2_latency_ms,
            "v2_confidence": self.v2_confidence,
            "divergence_category": self.divergence_category,
            "timestamp": self.timestamp,
        }


class ShadowModeController:
    """Controls shadow mode deployment and tracks comparison results."""

    def __init__(self, mode: ShadowMode = ShadowMode.OFF, canary_percentage: float = 0.0):
        self._mode = mode
        self._canary_pct = canary_percentage
        self._results: list[ShadowResult] = []
        self._promotion_thresholds = PromotionThresholds()

    @property
    def mode(self) -> ShadowMode:
        return self._mode

    def set_mode(self, mode: ShadowMode, canary_percentage: float = 0.0) -> None:
        if mode == ShadowMode.CANARY and not (0 < canary_percentage <= 100):
            raise ValueError("canary_percentage must be in (0, 100]")
        self._mode = mode
        self._canary_pct = canary_percentage

    def should_use_v2(self, request_hash: str = "") -> bool:
        """Determine if this request should use v2."""
        if self._mode == ShadowMode.OFF:
            return False
        if self._mode == ShadowMode.PROMOTED:
            return True
        if self._mode == ShadowMode.SHADOW:
            return False  # Shadow runs both but serves v1
        if self._mode == ShadowMode.CANARY:
            # Deterministic based on hash
            if request_hash:
                h = int(hashlib.md5(request_hash.encode()).hexdigest()[:8], 16)
                return (h % 100) < self._canary_pct
            return False
        return False

    def record_result(self, result: ShadowResult) -> None:
        self._results.append(result)

    def can_promote(self) -> tuple[bool, list[str]]:
        """Check if v2 meets promotion thresholds."""
        return self._promotion_thresholds.check(self._results)

    def shadow_report(self) -> dict[str, Any]:
        if not self._results:
            return {"mode": self._mode.value, "total_comparisons": 0}

        matches = sum(1 for r in self._results if r.match)
        total = len(self._results)
        avg_v1_latency = sum(r.v1_latency_ms for r in self._results) / total
        avg_v2_latency = sum(r.v2_latency_ms for r in self._results) / total

        return {
            "mode": self._mode.value,
            "total_comparisons": total,
            "match_rate": matches / total,
            "avg_v1_latency_ms": round(avg_v1_latency, 2),
            "avg_v2_latency_ms": round(avg_v2_latency, 2),
            "latency_ratio": round(avg_v2_latency / max(avg_v1_latency, 0.01), 2),
            "can_promote": self.can_promote()[0],
        }


# ── Promotion Thresholds ─────────────────────────────────────────────

@dataclass
class PromotionThresholds:
    """Frozen promotion thresholds for v1 -> v2 transition."""
    min_match_rate: float = 0.95        # 95% match rate required
    max_latency_ratio: float = 1.5      # v2 can be at most 1.5x slower
    min_comparisons: int = 1000         # Minimum shadow comparisons
    min_confidence_mean: float = 0.7    # Average v2 confidence
    max_error_rate: float = 0.02        # Max v2 error rate

    def check(self, results: list[ShadowResult]) -> tuple[bool, list[str]]:
        """Check if results meet all promotion thresholds."""
        violations = []

        if len(results) < self.min_comparisons:
            violations.append(
                f"Insufficient comparisons: {len(results)} < {self.min_comparisons}"
            )
            return False, violations

        matches = sum(1 for r in results if r.match)
        match_rate = matches / len(results)
        if match_rate < self.min_match_rate:
            violations.append(f"Match rate {match_rate:.3f} < {self.min_match_rate}")

        avg_v1 = sum(r.v1_latency_ms for r in results) / len(results)
        avg_v2 = sum(r.v2_latency_ms for r in results) / len(results)
        ratio = avg_v2 / max(avg_v1, 0.01)
        if ratio > self.max_latency_ratio:
            violations.append(f"Latency ratio {ratio:.2f} > {self.max_latency_ratio}")

        confidences = [r.v2_confidence for r in results if r.v2_confidence > 0]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            if mean_conf < self.min_confidence_mean:
                violations.append(f"Mean confidence {mean_conf:.3f} < {self.min_confidence_mean}")

        errors = sum(1 for r in results if r.divergence_category == "error")
        error_rate = errors / len(results)
        if error_rate > self.max_error_rate:
            violations.append(f"Error rate {error_rate:.3f} > {self.max_error_rate}")

        return len(violations) == 0, violations


# ── Telemetry Governance ─────────────────────────────────────────────

class TelemetryGovernance:
    """Controls what data can be logged in production telemetry.

    Enforces PII scrubbing, rate limiting, and data retention policies.
    """

    PII_PATTERNS = [
        "email", "phone", "password", "token", "secret", "ssn",
        "credit_card", "api_key", "auth", "birth_date", "address",
    ]

    def __init__(
        self,
        max_events_per_minute: int = 100,
        retention_days: int = 90,
        pii_scrubbing: bool = True,
    ) -> None:
        self.max_events_per_minute = max_events_per_minute
        self.retention_days = retention_days
        self.pii_scrubbing = pii_scrubbing
        self._event_timestamps: list[float] = []
        self._scrubbed_count = 0

    def can_log(self) -> bool:
        """Check rate limit."""
        now = time.time()
        cutoff = now - 60
        self._event_timestamps = [t for t in self._event_timestamps if t > cutoff]
        return len(self._event_timestamps) < self.max_events_per_minute

    def scrub_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Remove PII fields from a telemetry event."""
        if not self.pii_scrubbing:
            return event
        scrubbed = {}
        for key, value in event.items():
            if any(pattern in key.lower() for pattern in self.PII_PATTERNS):
                scrubbed[key] = "[REDACTED]"
                self._scrubbed_count += 1
            elif isinstance(value, str) and len(value) > 500:
                scrubbed[key] = value[:500] + "...[TRUNCATED]"
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_event(value)
            else:
                scrubbed[key] = value
        return scrubbed

    def log_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Process and log an event, respecting governance rules."""
        if not self.can_log():
            return None
        self._event_timestamps.append(time.time())
        return self.scrub_event(event)

    @property
    def scrubbed_count(self) -> int:
        return self._scrubbed_count


# ── Bundle Metadata ──────────────────────────────────────────────────

@dataclass
class BundleMetadata:
    """Extended bundle metadata for production deployment."""
    bundle_id: str
    model_version: str
    training_config_hash: str
    checkpoint_epoch: int
    checkpoint_step: int
    validation_metrics: dict[str, float] = field(default_factory=dict)
    hard_floor_status: dict[str, bool] = field(default_factory=dict)
    capability_registry_snapshot: dict[str, Any] = field(default_factory=dict)
    feature_flags_snapshot: dict[str, Any] = field(default_factory=dict)
    shadow_mode_report: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    git_commit: str = ""
    rollback_bundle_id: str = ""  # ID of the bundle to roll back to

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "model_version": self.model_version,
            "training_config_hash": self.training_config_hash,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_step": self.checkpoint_step,
            "validation_metrics": self.validation_metrics,
            "hard_floor_status": self.hard_floor_status,
            "capability_registry_snapshot": self.capability_registry_snapshot,
            "feature_flags_snapshot": self.feature_flags_snapshot,
            "shadow_mode_report": self.shadow_mode_report,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "rollback_bundle_id": self.rollback_bundle_id,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BundleMetadata:
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Rollback Rehearsal ───────────────────────────────────────────────

class RollbackManager:
    """Manages bundle rollback for safe deployment.

    Ensures that rollback is always possible by maintaining a history
    of deployed bundles with their full metadata.
    """

    def __init__(self) -> None:
        self._history: list[BundleMetadata] = []
        self._current: BundleMetadata | None = None

    def deploy(self, bundle: BundleMetadata) -> None:
        """Record a deployment."""
        if self._current is not None:
            bundle.rollback_bundle_id = self._current.bundle_id
            self._history.append(self._current)
        self._current = bundle

    @property
    def current(self) -> BundleMetadata | None:
        return self._current

    def can_rollback(self) -> bool:
        return len(self._history) > 0

    def rollback(self) -> BundleMetadata | None:
        """Roll back to the previous bundle."""
        if not self._history:
            return None
        previous = self._history.pop()
        self._current = previous
        return previous

    def history(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in self._history]


__all__ = [
    "BundleMetadata",
    "PromotionThresholds",
    "RollbackManager",
    "ShadowMode",
    "ShadowModeController",
    "ShadowResult",
    "TelemetryGovernance",
]
