"""Bundle manifest dataclass and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BundleManifest:
    """Describes a complete model bundle and its provenance."""

    bundle_id: str = ""
    status: str = "candidate"  # candidate | validated | failed | current
    created_at: str = ""
    git_commit: str = "unknown"
    pipeline_run_id: str = ""
    training_config_path: str = ""
    training_config_hash: str = ""
    datasets: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=lambda: {
        "retrieval_ir": "retrieval_ir/",
        "neural_ir": "neural_ir/",
        "adaptive_ranker": "adaptive_ranker/",
        "semantic_defaults": "semantic_defaults/",
        "evaluation": "evaluation/",
        "generic_training": "generic_training/",
        "configs": "configs/",
    })
    artifacts: dict[str, str] = field(default_factory=lambda: {
        "retrieval_manifest": "retrieval_ir/manifest.json",
        "neural_manifest": "neural_ir/manifest.json",
        "ranker_manifest": "adaptive_ranker/manifest.json",
        "dataset_contribution_report": "generic_training/dataset_contribution_report.json",
        "unsupported_sql_report": "generic_training/unsupported_sql_report.json",
    })
    metrics: dict[str, Any] = field(default_factory=lambda: {
        "query_ir_validity_rate": 0.0,
        "sql_validation_rate": 0.0,
        "unnecessary_join_rate": 0.0,
        "wrong_table_rate": 0.0,
        "unsafe_sql_count": 0,
    })
    classification_metrics: dict[str, Any] = field(default_factory=dict)
    confusion_matrices: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    percentiles: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    schema_drift_baseline: dict[str, Any] = field(default_factory=dict)
    statistical_promotion: dict[str, Any] = field(default_factory=dict)
    lifecycle_proof: dict[str, Any] = field(default_factory=dict)
    quality_gate: dict[str, Any] = field(default_factory=lambda: {
        "passed": False,
        "report_path": "evaluation/model_quality_gate_report.json",
    })
    pipeline_report: str = "pipeline/train_model_report.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "status": self.status,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "pipeline_run_id": self.pipeline_run_id,
            "training_config_path": self.training_config_path,
            "training_config_hash": self.training_config_hash,
            "datasets": self.datasets,
            "paths": self.paths,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "classification_metrics": self.classification_metrics,
            "confusion_matrices": self.confusion_matrices,
            "calibration": self.calibration,
            "percentiles": self.percentiles,
            "latency": self.latency,
            "schema_drift_baseline": self.schema_drift_baseline,
            "statistical_promotion": self.statistical_promotion,
            "lifecycle_proof": self.lifecycle_proof,
            "quality_gate": self.quality_gate,
            "pipeline_report": self.pipeline_report,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleManifest":
        return cls(
            bundle_id=data.get("bundle_id", ""),
            status=data.get("status", "candidate"),
            created_at=data.get("created_at", ""),
            git_commit=data.get("git_commit", "unknown"),
            pipeline_run_id=data.get("pipeline_run_id", ""),
            training_config_path=data.get("training_config_path", ""),
            training_config_hash=data.get("training_config_hash", ""),
            datasets=data.get("datasets", []),
            paths=data.get("paths", {}),
            artifacts=data.get("artifacts", {}),
            metrics=data.get("metrics", {}),
            classification_metrics=data.get("classification_metrics", {}),
            confusion_matrices=data.get("confusion_matrices", {}),
            calibration=data.get("calibration", {}),
            percentiles=data.get("percentiles", {}),
            latency=data.get("latency", {}),
            schema_drift_baseline=data.get("schema_drift_baseline", {}),
            statistical_promotion=data.get("statistical_promotion", {}),
            lifecycle_proof=data.get("lifecycle_proof", {}),
            quality_gate=data.get("quality_gate", {}),
            pipeline_report=data.get("pipeline_report", ""),
        )


def load_manifest(path: str | Path) -> BundleManifest:
    """Load a bundle manifest from a JSON file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    return BundleManifest.from_dict(data)


def save_manifest(manifest: BundleManifest, path: str | Path) -> None:
    """Save a bundle manifest to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
