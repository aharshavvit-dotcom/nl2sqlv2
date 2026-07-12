"""Tests for Gate 6: Production Readiness."""

from __future__ import annotations

import pytest

from deployment.production_readiness import (
    BundleMetadata,
    PromotionThresholds,
    RollbackManager,
    ShadowMode,
    ShadowModeController,
    ShadowResult,
    TelemetryGovernance,
)


class TestShadowMode:
    def test_off_mode_serves_v1(self):
        ctrl = ShadowModeController(ShadowMode.OFF)
        assert ctrl.should_use_v2("test_hash") is False

    def test_promoted_mode_serves_v2(self):
        ctrl = ShadowModeController(ShadowMode.PROMOTED)
        assert ctrl.should_use_v2("test_hash") is True

    def test_shadow_mode_runs_both(self):
        ctrl = ShadowModeController(ShadowMode.SHADOW)
        # Shadow runs both but serves v1
        assert ctrl.should_use_v2("test_hash") is False

    def test_canary_is_deterministic(self):
        ctrl = ShadowModeController()
        ctrl.set_mode(ShadowMode.CANARY, canary_percentage=50)
        result1 = ctrl.should_use_v2("consistent_hash")
        result2 = ctrl.should_use_v2("consistent_hash")
        assert result1 == result2

    def test_shadow_report(self):
        ctrl = ShadowModeController(ShadowMode.SHADOW)
        ctrl.record_result(ShadowResult(
            question="How many orders?",
            v1_sql="SELECT COUNT(*) FROM orders",
            v2_sql="SELECT COUNT(*) FROM orders",
            v1_latency_ms=10.0,
            v2_latency_ms=12.0,
            match=True,
            v2_confidence=0.95,
        ))
        report = ctrl.shadow_report()
        assert report["total_comparisons"] == 1
        assert report["match_rate"] == 1.0


class TestPromotionThresholds:
    def test_insufficient_data_fails(self):
        thresholds = PromotionThresholds(min_comparisons=1000)
        results = [ShadowResult(
            question="q", v1_sql="s", v2_sql="s",
            v1_latency_ms=10, v2_latency_ms=10,
            match=True, v2_confidence=0.9,
        )] * 100
        passed, violations = thresholds.check(results)
        assert passed is False
        assert any("Insufficient" in v for v in violations)

    def test_good_results_pass(self):
        thresholds = PromotionThresholds(min_comparisons=10)
        results = [ShadowResult(
            question=f"q{i}", v1_sql=f"s{i}", v2_sql=f"s{i}",
            v1_latency_ms=10, v2_latency_ms=12,
            match=True, v2_confidence=0.9,
        ) for i in range(20)]
        passed, violations = thresholds.check(results)
        assert passed is True


class TestTelemetryGovernance:
    def test_pii_scrubbing(self):
        gov = TelemetryGovernance()
        event = {"question": "test", "user_email": "a@b.com", "password_hash": "xyz"}
        scrubbed = gov.scrub_event(event)
        assert scrubbed["user_email"] == "[REDACTED]"
        assert scrubbed["password_hash"] == "[REDACTED]"
        assert scrubbed["question"] == "test"

    def test_rate_limiting(self):
        gov = TelemetryGovernance(max_events_per_minute=2)
        assert gov.log_event({"a": 1}) is not None
        assert gov.log_event({"a": 2}) is not None
        assert gov.log_event({"a": 3}) is None  # Rate limited


class TestRollbackManager:
    def test_deploy_and_rollback(self):
        mgr = RollbackManager()
        v1 = BundleMetadata(bundle_id="v1", model_version="1.0",
                           training_config_hash="abc", checkpoint_epoch=1, checkpoint_step=100)
        v2 = BundleMetadata(bundle_id="v2", model_version="2.0",
                           training_config_hash="def", checkpoint_epoch=2, checkpoint_step=200)
        mgr.deploy(v1)
        mgr.deploy(v2)
        assert mgr.current.bundle_id == "v2"
        assert mgr.can_rollback() is True

        rolled = mgr.rollback()
        assert rolled.bundle_id == "v1"
        assert mgr.current.bundle_id == "v1"

    def test_no_rollback_on_first_deploy(self):
        mgr = RollbackManager()
        v1 = BundleMetadata(bundle_id="v1", model_version="1.0",
                           training_config_hash="abc", checkpoint_epoch=1, checkpoint_step=100)
        mgr.deploy(v1)
        assert mgr.can_rollback() is False

    def test_rollback_bundle_id_set(self):
        mgr = RollbackManager()
        v1 = BundleMetadata(bundle_id="v1", model_version="1.0",
                           training_config_hash="abc", checkpoint_epoch=1, checkpoint_step=100)
        v2 = BundleMetadata(bundle_id="v2", model_version="2.0",
                           training_config_hash="def", checkpoint_epoch=2, checkpoint_step=200)
        mgr.deploy(v1)
        mgr.deploy(v2)
        assert mgr.current.rollback_bundle_id == "v1"


class TestBundleMetadata:
    def test_serialization_roundtrip(self):
        import tempfile
        import os
        bundle = BundleMetadata(
            bundle_id="test_001",
            model_version="2.0",
            training_config_hash="abc123",
            checkpoint_epoch=5,
            checkpoint_step=500,
            validation_metrics={"accuracy": 0.92},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            bundle.save(path)
            loaded = BundleMetadata.load(path)
            assert loaded.bundle_id == "test_001"
            assert loaded.validation_metrics["accuracy"] == 0.92
        finally:
            os.unlink(path)
