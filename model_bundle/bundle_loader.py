"""Load a validated model bundle for runtime use."""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any

from .bundle_manifest import BundleManifest, load_manifest
from .bundle_validator import ModelBundleValidator

logger = logging.getLogger(__name__)


def inspect_bundle_status(current_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    """Return user-facing production/candidate and quality-gate status."""
    current = Path(current_dir)
    candidate = Path(candidate_dir)
    gate_path = candidate / "evaluation" / "model_quality_gate_report.json"
    gate = {}
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            gate = {}
    blockers = gate.get("blocking_failures") or gate.get("failed_checks") or []
    blocker_names = [
        str(item.get("metric") if isinstance(item, dict) else item)
        for item in blockers
    ]
    return {
        "current_bundle_found": (current / "bundle_manifest.json").exists(),
        "candidate_bundle_found": (candidate / "bundle_manifest.json").exists(),
        "last_quality_gate_passed": gate.get("passed"),
        "top_blockers": blocker_names[:5],
        "candidate_debug_production_ready": False,
    }


class ModelBundleLoader:
    """Loads a validated model bundle, resolving all artifact paths."""

    def load(
        self,
        bundle_dir: str | Path,
        *,
        allow_candidate_debug: bool = False,
    ) -> dict[str, Any]:
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
        environment = str(os.getenv("NL2SQL_ENV", "development")).strip().lower()
        if environment not in {"development", "staging", "production"}:
            raise ValueError(f"Invalid NL2SQL_ENV={environment!r}; expected development, staging, or production.")
        if environment == "production" and manifest.status != "current":
            raise ValueError("Production mode forbids candidate bundle loading; use the current bundle.")
        if manifest.status == "candidate" and not allow_candidate_debug:
            raise ValueError(
                "Candidate bundle loading is disabled. Set "
                "NL2SQL_ALLOW_CANDIDATE_BUNDLE=1 only for explicit debugging."
            )
        validation = ModelBundleValidator().validate(
            path,
            allow_failed_quality_gate_debug=bool(
                manifest.status == "candidate" and allow_candidate_debug
            ),
        )
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
        if environment == "production":
            production_issues = []
            if manifest.quality_gate_mode not in {"production", "release"}:
                production_issues.append("quality_gate_mode is not production")
            if not manifest.quality_gate_passed or not (manifest.quality_gate or {}).get("passed", False):
                production_issues.append("quality gate did not pass")
            if not manifest.production_ready_full:
                production_issues.append("production_ready_full is false")
            if production_issues:
                raise ValueError("Production bundle is not deployable: " + "; ".join(production_issues))

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
            "bundle_source": "candidate_debug" if manifest.status == "candidate" else "current",
            "quality_gate_passed": bool((manifest.quality_gate or {}).get("passed", False)),
            "production_ready": bool((manifest.lifecycle_proof or {}).get("production_ready", False)),
            "production_ready_full": bool(manifest.production_ready_full),
            "quality_gate_mode": manifest.quality_gate_mode,
            "runtime_environment": environment,
            "loaded_for_debug": bool(manifest.status == "candidate" and allow_candidate_debug),
            "bundle_validation": validation,
            "neural_training_config": manifest.neural_training_config,
            "dataset_contribution_status": manifest.dataset_contribution_status,
            "sklearn_artifact_version": manifest.sklearn_artifact_version,
        }

        return result

    def load_preferred(
        self,
        current_dir: str | Path,
        candidate_dir: str | Path,
        *,
        allow_candidate_debug: bool = False,
    ) -> dict[str, Any]:
        """Load current first; candidate fallback requires an explicit debug flag."""
        try:
            return self.load(current_dir)
        except (FileNotFoundError, ValueError) as current_error:
            if not allow_candidate_debug:
                raise current_error
        return self.load(candidate_dir, allow_candidate_debug=True)
