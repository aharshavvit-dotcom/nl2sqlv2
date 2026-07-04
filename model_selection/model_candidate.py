from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelCandidate:
    name: str
    artifact_dir: str
    model_type: str
    metrics: dict[str, Any]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    model_artifact_source: str = "model_bundle"
    evaluation_mode: str = "real_model_predictions"
    eligible_for_promotion: bool = True
    candidate_bundle_id: str | None = None
    manifest_bundle_id: str | None = None
    pipeline_run_id: str | None = None
    generated_at: str | None = None
    report_path: str | None = None
