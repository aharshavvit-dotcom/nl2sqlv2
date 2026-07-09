"""Unit and integration tests for centralized promotion policy and recoverable swaps."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import pytest

from model_bundle.bundle_manifest import BundleManifest, save_manifest, load_manifest
from model_bundle.bundle_promoter import (
    PromotionPolicyEvaluator,
    PromotionRecoveryManager,
    ModelBundlePromoter,
)


@pytest.fixture
def base_candidate(tmp_path):
    """Create a mock candidate bundle folder structure."""
    cand = tmp_path / "candidate"
    cand.mkdir()
    
    # 1. Manifest
    manifest = BundleManifest(
        bundle_id="cand_123",
        status="candidate",
        datasets=["wikisql"],
        paths={
            "retrieval_ir": "retrieval_ir/",
            "evaluation": "evaluation/",
            "generic_training": "generic_training/",
            "configs": "configs/",
        },
        metrics={
            "unsafe_sql_count": 0,
            "sql_validation_rate": 1.0,
            "query_ir_validity_rate": 1.0,
            "unnecessary_join_rate": 0.0,
            "wrong_table_rate": 0.0,
        },
        quality_gate={"passed": True, "required": True, "report_path": "evaluation/model_quality_gate_report.json"},
        lifecycle_proof={
            "controlled_predicted_sql_report_attached_to_bundle": True,
            "controlled_predicted_sql_report_identity_verified": True,
            "controlled_predicted_sql_passed": True,
        },
        dataset_contribution_status={"passed": True},
        sklearn_artifact_version={"sklearn_version": "1.2.2"},
        eligible_for_promotion=True,
        production_ready_full=True,
        quality_gate_mode="production",
    )
    save_manifest(manifest, cand / "bundle_manifest.json")
    
    # 2. Subdirs
    (cand / "retrieval_ir").mkdir()
    (cand / "retrieval_ir" / "manifest.json").write_text(json.dumps({"sklearn_artifact_metadata": {"sklearn_version": "1.2.2"}}))
    (cand / "evaluation").mkdir()
    (cand / "evaluation" / "model_quality_gate_report.json").write_text(
        json.dumps({"passed": True, "eligible_for_promotion": True, "quality_gate_mode": "production"}),
        encoding="utf-8",
    )
    (cand / "evaluation" / "model_selection_report.json").write_text(
        json.dumps({
            "selection_blocked": False,
            "selected_model": "neural_v1",
            "candidate_bundle_id": "cand_123",
            "manifest_bundle_id": "cand_123",
            "selection_mode": "production",
        }),
        encoding="utf-8",
    )
    (cand / "evaluation" / "controlled_predicted_sql_execution_report.json").write_text("{}", encoding="utf-8")
    
    (cand / "generic_training").mkdir()
    (cand / "generic_training" / "dataset_contribution_report.json").write_text("{}", encoding="utf-8")
    (cand / "generic_training" / "unsupported_sql_report.json").write_text("{}", encoding="utf-8")
    
    (cand / "configs").mkdir()
    return cand


def test_promotion_policy_evaluator_approves_valid_candidate(base_candidate):
    evaluator = PromotionPolicyEvaluator()
    decision = evaluator.evaluate(base_candidate)
    
    assert decision["decision"] == "approved"
    assert (base_candidate / "promotion_decision.json").exists()


def test_promotion_policy_evaluator_rejects_unsafe_sql(base_candidate):
    # Modify manifest to report unsafe SQL
    manifest = BundleManifest(
        bundle_id="cand_123",
        status="candidate",
        metrics={"unsafe_sql_count": 1, "sql_validation_rate": 1.0},
        quality_gate={"passed": True, "required": True},
        eligible_for_promotion=True,
        production_ready_full=True,
    )
    save_manifest(manifest, base_candidate / "bundle_manifest.json")
    
    evaluator = PromotionPolicyEvaluator()
    decision = evaluator.evaluate(base_candidate)
    
    assert decision["decision"] == "rejected"
    assert "unsafe_sql_detected" in decision["blocking_issues"]


def test_promotion_policy_evaluator_rejects_low_sql_validation(base_candidate):
    # Modify manifest to report low validation rate
    manifest = load_manifest(base_candidate / "bundle_manifest.json")
    manifest.metrics["sql_validation_rate"] = 0.85
    save_manifest(manifest, base_candidate / "bundle_manifest.json")
    
    evaluator = PromotionPolicyEvaluator()
    decision = evaluator.evaluate(base_candidate)
    
    assert decision["decision"] == "rejected"
    assert any("sql_validation_rate_below_threshold" in issue for issue in decision["blocking_issues"])


def test_promotion_recovery_manager_locks_exclusively(tmp_path):
    lock_path = tmp_path / "promotion.lock"
    journal_path = tmp_path / "promotion_journal.json"
    
    mgr1 = PromotionRecoveryManager(lock_path, journal_path)
    mgr2 = PromotionRecoveryManager(lock_path, journal_path)
    
    mgr1.acquire_lock()
    
    with pytest.raises(RuntimeError, match="Could not acquire promotion lock"):
        mgr2.acquire_lock()
        
    mgr1.release_lock()
    # Now mgr2 should be able to acquire
    mgr2.acquire_lock()
    mgr2.release_lock()


def test_promotion_recovery_manager_journal_rollback(tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    (current_dir / "bundle_manifest.json").write_text('{"bundle_id": "current_active"}')
    
    backup_dir = tmp_path / "current.backup.123"
    backup_dir.mkdir()
    (backup_dir / "bundle_manifest.json").write_text('{"bundle_id": "backup_active"}')
    
    tmp_dir = tmp_path / "current.tmp.123"
    tmp_dir.mkdir()
    
    journal_path = tmp_path / "promotion_journal.json"
    lock_path = tmp_path / "promotion.lock"
    
    recovery_mgr = PromotionRecoveryManager(lock_path, journal_path)
    
    # Simulate failed swap at 'backed_up' stage
    recovery_mgr.write_journal({
        "stage": "backed_up",
        "candidate_id": "new_cand",
        "tmp_dir": str(tmp_dir),
        "backup_dir": str(backup_dir),
    })
    
    # Delete current dir to simulate interruption before current was restored
    shutil.rmtree(current_dir)
    
    recovery_mgr.recover(current_dir)
    
    # Assert current dir restored from backup_dir
    assert current_dir.exists()
    manifest = json.loads((current_dir / "bundle_manifest.json").read_text())
    assert manifest["bundle_id"] == "backup_active"
    
    # Assert temporary folder was cleaned up
    assert not tmp_dir.exists()
    assert not backup_dir.exists()
    assert not journal_path.exists()
