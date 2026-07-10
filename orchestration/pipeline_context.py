from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineContext:
    pipeline_run_id: str
    runtime_mode: str
    effective_config_hash: str
    split_version: str
    dataset_hashes: dict[str, str]
    git_commit: str
    dependency_fingerprint: dict[str, str]
    artifact_root: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_root"] = str(self.artifact_root)
        return payload
