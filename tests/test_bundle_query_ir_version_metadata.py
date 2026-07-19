"""
Purpose: Protects ir contract behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from model_bundle.bundle_manifest import BundleManifest


def test_bundle_manifest_declares_query_ir_version_policy_defaults() -> None:
    payload = BundleManifest().to_dict()

    assert payload["query_ir_versions_supported"] == ["1", "2.0"]
    assert payload["model_output_query_ir_version"] == "1"
    assert payload["runtime_preferred_query_ir_version"] == "1"


def test_legacy_bundle_manifest_defaults_to_phase2a_query_ir_policy() -> None:
    manifest = BundleManifest.from_dict({"bundle_id": "legacy"})

    assert manifest.query_ir_versions_supported == ["1", "2.0"]
    assert manifest.model_output_query_ir_version == "1"
    assert manifest.runtime_preferred_query_ir_version == "1"
