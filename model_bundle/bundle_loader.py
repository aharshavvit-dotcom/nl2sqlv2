"""Load a validated model bundle for runtime use."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .bundle_manifest import BundleManifest, load_manifest
from .bundle_validator import ModelBundleValidator

logger = logging.getLogger(__name__)


class ModelBundleLoader:
    """Loads a validated model bundle, resolving all artifact paths."""

    def load(self, bundle_dir: str | Path) -> dict[str, Any]:
        """Load a bundle and return resolved artifact paths.

        Rules:
            1. Refuse to load if manifest is missing.
            2. Refuse to load if bundle status is 'failed'.
            3. Warn if bundle status is 'candidate' (not yet promoted).
            4. Prefer 'current' bundle.
            5. Never guess legacy artifact folders during normal runtime.

        Returns:
            dict with keys: manifest, retrieval_model_dir, neural_model_dir,
            ranker_dir, semantic_defaults_dir, evaluation_dir
        """
        path = Path(bundle_dir)
        manifest_path = path / "bundle_manifest.json"

        # Rule 1: Refuse missing manifest
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No bundle manifest found at {manifest_path}. "
                "Run: python training/train_model.py --config configs/training.yaml"
            )

        manifest = load_manifest(manifest_path)
        validation = ModelBundleValidator().validate(path)
        if not validation.get("passed"):
            issues = validation.get("blocking_issues", [])
            raise ValueError(
                "Invalid model bundle:\n" + "\n".join(f"- {issue}" for issue in issues)
            )

        # Rule 2: Refuse failed bundle
        if manifest.status == "failed":
            raise ValueError(
                f"Bundle {manifest.bundle_id} has status 'failed'. "
                "Run training again to produce a valid bundle."
            )

        # Rule 3: Warn on candidate bundle
        if manifest.status == "candidate":
            logger.warning(
                "Loading candidate bundle %s — not yet promoted to 'current'. "
                "Quality gate may not have passed.", manifest.bundle_id
            )

        # Resolve paths from manifest
        evaluation_dir = str(path / manifest.paths.get("evaluation", "evaluation/"))
        calibration_report_path = str(Path(evaluation_dir) / "calibration_report.json")
        result: dict[str, Any] = {
            "manifest": manifest.to_dict(),
            "bundle_dir": str(path),
            "retrieval_model_dir": str(path / manifest.paths.get("retrieval_ir", "retrieval_ir/")),
            "neural_model_dir": str(path / manifest.paths.get("neural_ir", "neural_ir/")),
            "ranker_dir": str(path / manifest.paths.get("adaptive_ranker", "adaptive_ranker/")),
            "semantic_defaults_dir": str(path / manifest.paths.get("semantic_defaults", "semantic_defaults/")),
            "evaluation_dir": evaluation_dir,
            "calibration_dir": evaluation_dir,
            "calibration_report_path": calibration_report_path if Path(calibration_report_path).exists() else None,
        }

        return result
