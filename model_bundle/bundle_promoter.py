"""Promote a candidate bundle to current after validation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle_manifest import load_manifest, save_manifest
from .bundle_validator import ModelBundleValidator


logger = logging.getLogger(__name__)


class PromotionPolicyEvaluator:
    """Evaluates candidate bundle evidence and returns a signed decision."""

    def __init__(self) -> None:
        pass

    def evaluate(self, candidate_dir: Path) -> dict[str, Any]:
        """Verify quality policies and independent blocking gates."""
        manifest = load_manifest(candidate_dir / "bundle_manifest.json")
        
        gate_path = candidate_dir / "evaluation" / "model_quality_gate_report.json"
        selection_path = candidate_dir / "evaluation" / "model_selection_report.json"
        controlled_path = candidate_dir / "evaluation" / "controlled_predicted_sql_execution_report.json"
        
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        lifecycle = manifest.lifecycle_proof or {}
        
        blockers: list[str] = []
        
        # 1. Independent validation & safety thresholds
        metrics = manifest.metrics or {}
        unsafe_sql = metrics.get("unsafe_sql_count", 0)
        sql_val = metrics.get("sql_validation_rate", 0.0)
        wrong_table = metrics.get("wrong_table_rate", 0.0)
        unnecessary_join = metrics.get("unnecessary_join_rate", 0.0)
        
        if unsafe_sql > 0 or gate.get("unsafe_sql_count", 0) > 0:
            blockers.append("unsafe_sql_detected")
        if sql_val < 0.90:
            blockers.append(f"sql_validation_rate_below_threshold: {sql_val:.2f} < 0.90")
        if wrong_table > 0.15:
            blockers.append(f"wrong_table_rate_above_threshold: {wrong_table:.2f} > 0.15")
        if unnecessary_join > 0.05:
            blockers.append(f"unnecessary_join_rate_above_threshold: {unnecessary_join:.2f} > 0.05")
            
        # 2. Manifest status and readiness checks
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
        if manifest.dataset_contribution_status.get("passed") is not True:
            blockers.append("dataset_contribution_invalid")
        if not manifest.sklearn_artifact_version.get("sklearn_version"):
            blockers.append("sklearn_artifact_version_missing")
            
        # 3. Report identity validation
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
                "decision": "rejected",
                "blocking_issues": blockers,
            }
            
        # Write decision file inside candidate
        decision_report = {
            "decision": "approved",
            "pipeline_run_id": manifest.pipeline_run_id,
            "bundle_id": manifest.bundle_id,
            "candidate_sha256": self._compute_dir_sha256(candidate_dir),
            "policy_version": "1.0",
            "approved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "checks": {
                "unsafe_sql_count_zero": bool(unsafe_sql == 0),
                "sql_validation_passed": bool(sql_val >= 0.90),
                "wrong_table_rate_valid": bool(wrong_table <= 0.15),
                "unnecessary_join_rate_valid": bool(unnecessary_join <= 0.05),
                "quality_gate_passed": True,
                "sklearn_metadata_present": True,
                "controlled_predicted_sql_passed": True,
            }
        }
        
        decision_path = candidate_dir / "promotion_decision.json"
        decision_path.write_text(json.dumps(decision_report, indent=2, ensure_ascii=False), encoding="utf-8")
        
        return decision_report

    def _compute_dir_sha256(self, dir_path: Path) -> str:
        hasher = hashlib.sha256()
        for path in sorted(dir_path.rglob("*")):
            if path.is_file() and path.name != "promotion_decision.json":
                hasher.update(path.relative_to(dir_path).as_posix().encode())
                try:
                    with path.open("rb") as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)
                except Exception:
                    pass
        return hasher.hexdigest()


class PromotionRecoveryManager:
    """Manages transactional recovery and locks for current bundle promotion."""

    def __init__(self, lock_path: Path, journal_path: Path) -> None:
        self.lock_path = lock_path
        self.journal_path = journal_path

    def acquire_lock(self) -> None:
        """Acquire exclusive promotion lock."""
        # Simple lock file directory check to ensure Windows concurrency safety
        for _ in range(10):
            try:
                self.lock_path.mkdir(parents=True, exist_ok=False)
                # Write PID
                (self.lock_path / "lock.pid").write_text(str(os.getpid()))
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("Could not acquire promotion lock. Another promotion is in progress.")

    def release_lock(self) -> None:
        """Release the lock."""
        if self.lock_path.exists():
            try:
                shutil.rmtree(self.lock_path)
            except Exception:
                pass

    def write_journal(self, state: dict[str, Any]) -> None:
        """Atomically write transition journal state."""
        tmp = self.journal_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if self.journal_path.exists():
            self.journal_path.unlink()
        tmp.rename(self.journal_path)

    def recover(self, current_dir: Path) -> None:
        """Check for unfinished journal and recover."""
        if not self.journal_path.exists():
            return
        
        try:
            journal = json.loads(self.journal_path.read_text(encoding="utf-8"))
            stage = journal.get("stage")
            backup_dir_str = journal.get("backup_dir")
            tmp_dir_str = journal.get("tmp_dir")
            
            logger.warning("Unfinished promotion journal detected! Restoring state from stage %s.", stage)
            
            # Stage A: swap was interrupted before current directory swap or renaming
            if stage in {"started", "copied_tmp"}:
                # Clean up tmp dir if it exists
                if tmp_dir_str and Path(tmp_dir_str).exists():
                    shutil.rmtree(tmp_dir_str)
                    
            # Stage B: backup created but swap interrupted/incomplete
            elif stage == "backed_up":
                # Swap back backup directory to current if it exists
                if backup_dir_str and Path(backup_dir_str).exists():
                    if current_dir.exists():
                        shutil.rmtree(current_dir)
                    shutil.move(backup_dir_str, str(current_dir))
                    
                if tmp_dir_str and Path(tmp_dir_str).exists():
                    shutil.rmtree(tmp_dir_str)
                    
            # Stage C: swapped but not fully validated/manifest saved
            elif stage == "swapped":
                # Current exists. Ensure validation passes or rollback
                manifest_path = current_dir / "bundle_manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = load_manifest(manifest_path)
                        manifest.status = "current"
                        manifest.bundle_status = "current"
                        manifest.model_artifact_source = "model_bundle_current"
                        save_manifest(manifest, manifest_path)
                    except Exception:
                        # Rollback if manifest saving fails
                        if backup_dir_str and Path(backup_dir_str).exists():
                            if current_dir.exists():
                                shutil.rmtree(current_dir)
                            shutil.move(backup_dir_str, str(current_dir))
                else:
                    if backup_dir_str and Path(backup_dir_str).exists():
                        if current_dir.exists():
                            shutil.rmtree(current_dir)
                        shutil.move(backup_dir_str, str(current_dir))
                        
            # Clean up leftover backup directory if completed
            if stage == "completed":
                if backup_dir_str and Path(backup_dir_str).exists():
                    shutil.rmtree(backup_dir_str)
                    
        except Exception as exc:
            logger.error("Failed to recover promotion journal: %s", exc)
        finally:
            if self.journal_path.exists():
                try:
                    self.journal_path.unlink()
                except Exception:
                    pass


class ModelBundlePromoter:
    """Promotes a validated candidate bundle to become the current active bundle."""

    def promote(
        self,
        candidate_dir: str | Path,
        current_dir: str | Path,
        skip_quality_gate: bool = False,
    ) -> dict[str, Any]:
        """Promotes candidate bundle using a recoverable swap flow.

        Returns:
            dict with promotion report.
        """
        candidate = Path(candidate_dir)
        current = Path(current_dir)
        
        # 1. Evaluate candidate
        evaluator = PromotionPolicyEvaluator()
        decision = evaluator.evaluate(candidate)
        if decision.get("decision") != "approved":
            return {
                "promoted": False,
                "reason": "Candidate bundle failed validation or promotion policies",
                "blocking_issues": decision.get("blocking_issues", []),
            }
            
        # 2. Acquire lock & initialize recovery manager
        lock_path = current.parent / "promotion.lock"
        journal_path = current.parent / "promotion_journal.json"
        recovery_mgr = PromotionRecoveryManager(lock_path, journal_path)
        
        recovery_mgr.acquire_lock()
        
        try:
            # Check for recovery on startup/interrupted state
            recovery_mgr.recover(current)
            
            # Swapping setup
            run_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            current_tmp = current.parent / f"current.tmp.{run_suffix}"
            backup_dir = current.parent / "history" / f"current.backup.{run_suffix}"
            
            if current_tmp.exists():
                shutil.rmtree(current_tmp)
                
            # Write initial state
            recovery_mgr.write_journal({
                "stage": "started",
                "candidate_id": load_manifest(candidate / "bundle_manifest.json").bundle_id,
                "tmp_dir": str(current_tmp),
                "backup_dir": str(backup_dir),
            })
            
            # Step A: Copy candidate to temp directory
            shutil.copytree(candidate, current_tmp)
            recovery_mgr.write_journal({
                "stage": "copied_tmp",
                "candidate_id": load_manifest(candidate / "bundle_manifest.json").bundle_id,
                "tmp_dir": str(current_tmp),
                "backup_dir": str(backup_dir),
            })
            
            # Step B: Backup old current
            renamed_backup = False
            if current.exists():
                backup_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), str(backup_dir))
                renamed_backup = True
                
            recovery_mgr.write_journal({
                "stage": "backed_up",
                "candidate_id": load_manifest(candidate / "bundle_manifest.json").bundle_id,
                "tmp_dir": str(current_tmp),
                "backup_dir": str(backup_dir),
            })
            
            # Step C: Move tmp to current
            shutil.move(str(current_tmp), str(current))
            recovery_mgr.write_journal({
                "stage": "swapped",
                "candidate_id": load_manifest(candidate / "bundle_manifest.json").bundle_id,
                "tmp_dir": str(current_tmp),
                "backup_dir": str(backup_dir),
            })
            
            # Step D: Update manifest status and save
            manifest = load_manifest(current / "bundle_manifest.json")
            manifest.status = "current"
            manifest.bundle_status = "current"
            manifest.model_artifact_source = "model_bundle_current"
            save_manifest(manifest, current / "bundle_manifest.json")
            
            # Step E: Post-promotion validation check
            validator = ModelBundleValidator()
            validation = validator.validate(current)
            if not validation.get("passed"):
                # Rollback if loaded current validator fails
                if renamed_backup:
                    if current.exists():
                        shutil.rmtree(current)
                    shutil.move(str(backup_dir), str(current))
                raise ValueError("Promoted bundle failed post-swap validation check: " + ", ".join(validation.get("blocking_issues", [])))
                
            # Done!
            recovery_mgr.write_journal({
                "stage": "completed",
                "candidate_id": manifest.bundle_id,
                "tmp_dir": str(current_tmp),
                "backup_dir": str(backup_dir),
            })
            
            # Delete backup dir after successful check
            if renamed_backup and backup_dir.exists():
                shutil.rmtree(backup_dir)
                
            # Write final promotion report
            promotion_report = {
                "promoted": True,
                "bundle_id": manifest.bundle_id,
                "pipeline_run_id": manifest.pipeline_run_id,
                "promoted_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "candidate_dir": str(candidate),
                "current_dir": str(current),
                "quality_gate_passed": True,
                "validation": validation,
            }
            
            report_path = current / "promotion_report.json"
            report_path.write_text(json.dumps(promotion_report, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Clean journal
            if journal_path.exists():
                journal_path.unlink()
                
            return promotion_report
            
        except Exception as exc:
            # Try atomic recovery rollback
            try:
                recovery_mgr.recover(current)
            except Exception:
                pass
            return {
                "promoted": False,
                "reason": f"Windows-safe promotion failed during folder swap: {exc}",
                "blocking_issues": [str(exc)],
            }
        finally:
            recovery_mgr.release_lock()
