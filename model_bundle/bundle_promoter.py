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
            2. Candidate must pass the production quality gate and identity checks.
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

        # 2. Promotion is never bypassable. Debug/baseline runs only build candidates.
        gate_path = candidate / "evaluation" / "model_quality_gate_report.json"
        selection_path = candidate / "evaluation" / "model_selection_report.json"
        controlled_path = candidate / "evaluation" / "controlled_predicted_sql_execution_report.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        lifecycle = manifest.lifecycle_proof or {}
        blockers: list[str] = []
        if skip_quality_gate:
            blockers.append("quality_gate_bypass_forbidden")
        if manifest.status != "candidate":
            blockers.append("bundle_status_not_candidate")
        if gate.get("quality_gate_mode") not in {"production", "release"}:
            blockers.append("quality_gate_mode_not_production")
        if gate.get("passed") is not True or manifest.quality_gate.get("passed") is not True:
            blockers.append("production_quality_gate_not_passed")
        if gate.get("eligible_for_promotion") is not True or manifest.eligible_for_promotion is not True:
            blockers.append("candidate_not_eligible_for_promotion")
        if manifest.production_ready_full is not True:
            blockers.append("production_ready_full_false")
        required_neural = {
            "epochs": 10,
            "batch_size": 8,
            "save_best_metric": "loss",
            "save_best_mode": "min",
            "early_stopping_patience": 2,
            "weight_decay": 0.0001,
            "pointer_head_weight_decay": 0.001,
            "pointer_dropout": 0.30,
        }
        if any(manifest.neural_training_config.get(key) != value for key, value in required_neural.items()):
            blockers.append("effective_neural_config_invalid")
        if manifest.dataset_contribution_status.get("passed") is not True:
            blockers.append("dataset_contribution_invalid")
        if not manifest.sklearn_artifact_version.get("sklearn_version"):
            blockers.append("sklearn_artifact_version_missing")
        if not controlled_path.exists() or not lifecycle.get("controlled_predicted_sql_report_attached_to_bundle", False):
            blockers.append("controlled_predicted_sql_report_missing")
        if not lifecycle.get("controlled_predicted_sql_report_identity_verified", False):
            blockers.append("controlled_predicted_sql_report_identity_invalid")
        if lifecycle.get("controlled_predicted_sql_report_identity_stale", False):
            blockers.append("controlled_predicted_sql_report_stale")
        if not selection:
            blockers.append("model_selection_report_missing")
        else:
            if selection.get("selection_blocked") is True or selection.get("selected_model") is None:
                blockers.append("model_selection_blocked")
            if selection.get("model_selection_stale") is True:
                blockers.append("model_selection_report_stale")
            if selection.get("selection_mode") != "production":
                blockers.append("model_selection_mode_not_production")
            if selection.get("required_metric_failures"):
                blockers.append("model_selection_required_metrics_missing")
            if selection.get("candidate_bundle_id") != manifest.bundle_id:
                blockers.append("model_selection_bundle_id_mismatch")
            if selection.get("manifest_bundle_id") != manifest.bundle_id:
                blockers.append("model_selection_manifest_id_mismatch")
        if lifecycle.get("controlled_predicted_sql_passed") is not True:
            blockers.append("controlled_predicted_sql_not_passed")
        if blockers:
            return {
                "promoted": False,
                "reason": "Production promotion requirements not met",
                "bundle_id": manifest.bundle_id,
                "blocking_issues": list(dict.fromkeys(blockers)),
                "quality_gate": gate or manifest.quality_gate,
            }

        # 3. Windows-safe atomic promotion flow
        import uuid
        run_suffix = manifest.pipeline_run_id or uuid.uuid4().hex[:8]
        current_tmp = current.parent / f"current.tmp.{run_suffix}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = history_base / f"current.backup.{timestamp}"

        # Clean up any leftover temp dirs
        if current_tmp.exists():
            try:
                shutil.rmtree(current_tmp)
            except Exception:
                pass

        try:
            # Step A: Copy candidate to temp directory first
            shutil.copytree(candidate, current_tmp)
            
            # Step B: Rename existing current directory to backup
            renamed_backup = False
            if current.exists():
                backup_dir.parent.mkdir(parents=True, exist_ok=True)
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.move(str(current), str(backup_dir))
                renamed_backup = True

            # Step C: Rename temp directory to current
            try:
                shutil.move(str(current_tmp), str(current))
            except Exception as exc:
                # Rollback if Step C fails
                if renamed_backup:
                    shutil.move(str(backup_dir), str(current))
                raise exc

            # Step D: Update manifest status to 'current' and save
            manifest.status = "current"
            manifest.bundle_status = "current"
            manifest.model_artifact_source = "model_bundle_current"
            save_manifest(manifest, current / "bundle_manifest.json")

            # Step E: Delete backup only after successful manifest save
            if renamed_backup and backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except Exception:
                    pass  # Advisory, don't fail promotion if backup cleanup fails

        except Exception as exc:
            # Final cleanup of temp directory if it still exists
            if current_tmp.exists():
                try:
                    shutil.rmtree(current_tmp)
                except Exception:
                    pass
            return {
                "promoted": False,
                "reason": f"Windows-safe promotion failed during folder swap: {exc}",
                "blocking_issues": [str(exc)],
            }

        # 6. Write promotion report
        promotion_report = {
            "promoted": True,
            "bundle_id": manifest.bundle_id,
            "pipeline_run_id": manifest.pipeline_run_id,
            "promoted_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "candidate_dir": str(candidate),
            "current_dir": str(current),
            "quality_gate_passed": manifest.quality_gate.get("passed", False),
            "quality_gate_skipped": False,
            "validation": validation,
        }
        report_path = current / "promotion_report.json"
        report_path.write_text(json.dumps(promotion_report, indent=2, ensure_ascii=False), encoding="utf-8")

        return promotion_report
