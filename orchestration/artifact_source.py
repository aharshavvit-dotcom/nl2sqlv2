"""Model artifact source enum for tracking evaluation context.

This module defines where model artifacts are loaded from at different
stages of the pipeline:
- WORK_ARTIFACTS: Pre-bundle — fresh outputs from the current training run
- CANDIDATE_BUNDLE: Post-bundle — candidate under review
- CURRENT_BUNDLE: Production runtime — promoted immutable bundle
"""

from __future__ import annotations

from enum import Enum


class ModelArtifactSource(str, Enum):
    """Where model artifacts are loaded from during evaluation/inference."""

    WORK_ARTIFACTS = "work_artifacts"
    CANDIDATE_BUNDLE = "candidate_bundle"
    CURRENT_BUNDLE = "current_bundle"

    @classmethod
    def from_string(cls, value: str) -> "ModelArtifactSource":
        """Parse a string to a ModelArtifactSource, with legacy fallback."""
        normalized = str(value or "").lower().strip()
        # Legacy mappings
        legacy_map = {
            "model_bundle": cls.CURRENT_BUNDLE,
            "model_bundle_candidate": cls.CANDIDATE_BUNDLE,
            "artifact_dirs": cls.WORK_ARTIFACTS,
            "legacy_cache": cls.WORK_ARTIFACTS,
            "none": cls.WORK_ARTIFACTS,
        }
        if normalized in legacy_map:
            return legacy_map[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.WORK_ARTIFACTS

    def is_pre_bundle(self) -> bool:
        """True if this source is from the current training run (pre-bundle)."""
        return self == self.WORK_ARTIFACTS

    def is_bundle(self) -> bool:
        """True if this source is from a bundle (candidate or current)."""
        return self in {self.CANDIDATE_BUNDLE, self.CURRENT_BUNDLE}
