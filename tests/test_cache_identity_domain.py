from __future__ import annotations

from inference.prediction_cache import BundleIdentity, PredictionCache


def test_cache_key_changes_with_bundle_identity_and_security_context(tmp_path) -> None:
    cache = PredictionCache(cache_path=tmp_path / "cache.db")
    schema = {
        "schema_fingerprint": "schema-a",
        "tenant_id": "tenant-a",
        "environment": "production",
        "database_role": "reader",
        "security_context_fingerprint": "rls-a",
    }
    identity_a = BundleIdentity("bundle-a", "ckpt-a", "retr-a", "route-a")
    identity_b = BundleIdentity("bundle-a", "ckpt-b", "retr-a", "route-a")

    key_a = cache.generate_hash_key("show orders", schema, None, bundle_identity=identity_a)
    key_b = cache.generate_hash_key("show orders", schema, None, bundle_identity=identity_b)
    key_c = cache.generate_hash_key(
        "show orders",
        {**schema, "security_context_fingerprint": "rls-b"},
        None,
        bundle_identity=identity_a,
    )

    assert key_a != key_b
    assert key_a != key_c


def test_production_cache_disabled_without_security_context(tmp_path) -> None:
    cache = PredictionCache(cache_path=tmp_path / "cache.db")
    schema = {"schema_fingerprint": "schema-a", "tenant_id": "tenant-a", "environment": "production"}
    prediction = {"status": "completed", "query_ir": {"intent": "show"}, "validation": {"is_valid": True}}

    cache.put("show orders", schema, None, prediction)

    assert cache.get("show orders", schema, None) is None
