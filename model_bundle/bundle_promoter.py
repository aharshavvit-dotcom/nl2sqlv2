"""Promote a candidate bundle to current after validation."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle_manifest import load_manifest, save_manifest
from .bundle_validator import ModelBundleValidator


class ModelBundlePromoter:
    """Promotes a validated candidate bundle to become the current active bundle."""

    def promote(
        self,
        candidate_dir: str | Path,
        current_dir: str | Path,
        skip_quality_gate: bool = False,
    ) -> dict[str, Any]:
        """Promote a candidate bundle to current.

        Rules:
            1. Validate candidate first.
            2. Candidate must pass quality gate unless explicitly skipped (smoke mode).
            3. Backup old current bundle to history.
            4. Promote candidate to current.
            5. Update manifest status to 'current'.
            6. Write promotion report.

        Returns:
            dict with promotion result including status and paths.
        """
        candidate = Path(candidate_dir)
        current = Path(current_dir)
        history_base = candidate.parent / "history"

        # 1. Validate candidate
        validator = ModelBundleValidator()
        validation = validator.validate(candidate)
        if not validation["passed"]:
            return {
                "promoted": False,
                "reason": "Candidate bundle failed validation",
                "blocking_issues": validation["blocking_issues"],
                "warnings": validation["warnings"],
            }

        # Load manifest
        manifest = load_manifest(candidate / "bundle_manifest.json")

        # 2. Check quality gate
        if not skip_quality_gate:
            qg_passed = manifest.quality_gate.get("passed", False)
            if not qg_passed:
                return {
                    "promoted": False,
                    "reason": "Quality gate not passed",
                    "bundle_id": manifest.bundle_id,
                    "quality_gate": manifest.quality_gate,
                }

        # 3. Backup old current bundle
        if current.exists() and (current / "bundle_manifest.json").exists():
            try:
                old_manifest = load_manifest(current / "bundle_manifest.json")
                backup_name = old_manifest.bundle_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            except Exception:
                backup_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_dir = history_base / backup_name
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(current, backup_dir)

        # 4. Promote candidate to current
        if current.exists():
            shutil.rmtree(current)
        shutil.copytree(candidate, current)

        # 5. Update manifest status to 'current'
        manifest.status = "current"
        save_manifest(manifest, current / "bundle_manifest.json")

        # 6. Write promotion report
        promotion_report = {
            "promoted": True,
            "bundle_id": manifest.bundle_id,
            "promoted_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "candidate_dir": str(candidate),
            "current_dir": str(current),
            "quality_gate_passed": manifest.quality_gate.get("passed", False),
            "quality_gate_skipped": skip_quality_gate,
            "validation": validation,
        }
        report_path = current / "promotion_report.json"
        report_path.write_text(json.dumps(promotion_report, indent=2, ensure_ascii=False), encoding="utf-8")

        return promotion_report
