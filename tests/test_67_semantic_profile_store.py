"""
Purpose: Protects general unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

from semantic_layer import build_semantic_profile
from semantic_layer.semantic_profile_store import SemanticProfileStore
from tests.fixtures.generic_schema import generic_schema


def test_semantic_profile_store_round_trips_by_fingerprint(tmp_path: Path) -> None:
    profile = build_semantic_profile(generic_schema())
    store = SemanticProfileStore(tmp_path)

    store.save(profile["schema_fingerprint"], profile)
    loaded = store.load(profile["schema_fingerprint"])

    assert loaded is not None
    assert loaded["schema_fingerprint"] == profile["schema_fingerprint"]
    assert store.list_profiles()[0]["table_count"] == 3
