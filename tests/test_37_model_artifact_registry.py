"""
Purpose: Protects execution unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_registry.artifact_registry import ArtifactRegistry


def test_artifact_registry_saves_manifest_and_finds_latest_passing(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    artifact_dir = tmp_path / "neural_ir_model"
    registry.register_model(
        artifact_dir,
        {
            "model_name": "neural_ir_model",
            "model_version": "2026-06-16_001",
            "created_at": "2026-06-16T00:00:00+00:00",
            "training_data": "data/processed/generic_ir_train.jsonl",
            "validation_data": "data/processed/generic_ir_validation.jsonl",
            "git_commit": "abc123",
            "metrics": {"sql_validation_rate": 1.0},
            "quality_gate": {"passed": True},
            "notes": "",
        },
    )

    manifest_path = artifact_dir / "model_manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["model_name"] == "neural_ir_model"
    assert registry.get_latest_passing_model("neural_ir_model")["artifact_dir"] == str(artifact_dir)
