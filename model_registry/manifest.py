from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .versioning import generate_model_version


class ModelManifest(BaseModel):
    model_name: str
    model_version: str = Field(default_factory=generate_model_version)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    training_data: str | None = None
    validation_data: str | None = None
    git_commit: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    quality_gate: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
