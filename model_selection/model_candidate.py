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
