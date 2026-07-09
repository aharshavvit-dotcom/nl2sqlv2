"""Run-scoped pipeline state management.

Provides a PipelineRunState that tracks what happened during a single evaluation
pipeline execution. This replaces ad-hoc state tracking scattered across the
evaluator, quality gate, and promotion modules.

Review Comment #8: Add run-scoped pipeline state.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evaluation.report_schemas import (
    REPORT_SCHEMA_VERSION,
    METRIC_DEFINITIONS_VERSION,
    ReportIdentity,
    RowAccounting,
    PromotionEligibility,
    CheckpointIdentity,
)


class PipelineRunState(BaseModel):
    """Immutable state for a single pipeline evaluation run.

    This is the single source of truth for what happened during the run.
    All downstream consumers (quality gate, promotion policy, report writer)
    read from this state rather than maintaining their own copies.
    """

    # --- Identity ---
    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    pipeline_name: str = ""
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    completed_at: str | None = None

    # --- Report identity ---
    report_identity: ReportIdentity = Field(default_factory=lambda: ReportIdentity(report_type="pipeline_run"))

    # --- Row accounting ---
    row_accounting: RowAccounting = Field(default_factory=RowAccounting)

    # --- Checkpoint identity ---
    checkpoint: CheckpointIdentity = Field(default_factory=CheckpointIdentity)

    # --- Promotion eligibility evidence ---
    promotion: PromotionEligibility = Field(default_factory=PromotionEligibility)

    # --- Artifacts produced ---
    artifacts_produced: list[str] = Field(default_factory=list)
    artifact_checksums: dict[str, str] = Field(default_factory=dict)

    # --- Quality gate result ---
    quality_gate_passed: bool | None = None
    quality_gate_mode: str | None = None
    quality_gate_failures: list[dict[str, Any]] = Field(default_factory=list)

    # --- Metadata ---
    model_artifact_source: str = "unknown"
    evaluation_mode: str = "unknown"
    full_bundle_runtime_used: bool = False
    calibration_loaded: bool = False

    def complete(self) -> "PipelineRunState":
        """Mark the run as completed."""
        self.completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return self

    def record_artifact(self, path: str | Path, content_hash: str | None = None) -> None:
        """Record an artifact produced during this run."""
        path_str = str(path)
        if path_str not in self.artifacts_produced:
            self.artifacts_produced.append(path_str)
        if content_hash:
            self.artifact_checksums[path_str] = content_hash

    def record_quality_gate(self, result: dict[str, Any]) -> None:
        """Record quality gate evaluation result."""
        self.quality_gate_passed = result.get("passed")
        self.quality_gate_mode = result.get("quality_gate_mode")
        self.quality_gate_failures = result.get("failed_checks", [])

    def to_report_envelope(self) -> dict[str, Any]:
        """Produce a report envelope that wraps this run's state.

        This envelope can be included at the top level of any report
        produced by this run.
        """
        return {
            "pipeline_run_state": self.model_dump(),
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
            "pipeline_run_id": self.run_id,
            "generated_at": self.completed_at or self.started_at,
        }

    def save(self, path: Path) -> None:
        """Save run state to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.model_dump(), indent=2, default=str)
        path.write_text(content, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineRunState":
        """Load run state from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_run_state(
    pipeline_name: str = "",
    pipeline_run_id: str | None = None,
    model_artifact_source: str = "unknown",
    evaluation_mode: str = "unknown",
    full_bundle_runtime_used: bool = False,
) -> PipelineRunState:
    """Factory function to create a new pipeline run state."""
    run_id = pipeline_run_id or f"run-{uuid.uuid4().hex[:12]}"
    return PipelineRunState(
        run_id=run_id,
        pipeline_name=pipeline_name,
        report_identity=ReportIdentity(
            report_type="pipeline_run",
            pipeline_run_id=run_id,
        ),
        model_artifact_source=model_artifact_source,
        evaluation_mode=evaluation_mode,
        full_bundle_runtime_used=full_bundle_runtime_used,
        calibration_loaded=full_bundle_runtime_used,
    )
