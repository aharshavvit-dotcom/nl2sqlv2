"""Unit and concurrency tests for the SQLite-backed prediction cache."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
import pytest

from inference.prediction_cache import PredictionCache


@pytest.fixture
def cache_db(tmp_path):
    db_path = tmp_path / "cache.db"
    return PredictionCache(cache_path=db_path, max_entries=3, ttl_days=1)


def test_cache_key_partitioning(cache_db):
    question = "show sales above 500"
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    
    # Base key
    key1 = cache_db.generate_hash_key(question, schema, None, None)
    
    # Different tenant_id
    schema_diff_tenant = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_B"}
    key2 = cache_db.generate_hash_key(question, schema_diff_tenant, None, None)
    assert key1 != key2
    
    # Different schema fingerprint
    schema_diff_sch = {"schema_fingerprint": "sch_456", "tenant_id": "tenant_A"}
    key3 = cache_db.generate_hash_key(question, schema_diff_sch, None, None)
    assert key1 != key3


def test_cache_put_and_get(cache_db):
    question = "show transactions"
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    
    prediction = {
        "status": "completed",
        "query_ir": {
            "intent": "show_records",
            "filters": [{"column": "amount", "value": 500}]
        },
        "validation": {"is_valid": True},
        "sql": "SELECT * FROM transactions WHERE amount > 500",
        "confidence": 0.9,
    }
    
    cache_db.put(question, schema, None, prediction)
    
    cached = cache_db.get(question, schema, None)
    assert cached is not None
    assert cached["status"] == "completed"
    # Question text should not be present in retrieved cached prediction
    assert "question" not in cached
    # Filter values should be redacted in cache query_ir
    assert cached["query_ir"]["filters"][0]["value"] == "[REDACTED]"


def test_cache_evicts_lru_and_enforces_capacity(cache_db):
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    
    prediction = {
        "status": "completed",
        "query_ir": {"intent": "show"},
        "validation": {"is_valid": True},
    }
    
    # Put 4 items (capacity is 3)
    cache_db.put("q1", schema, None, prediction)
    time.sleep(0.01)
    cache_db.put("q2", schema, None, prediction)
    time.sleep(0.01)
    cache_db.put("q3", schema, None, prediction)
    time.sleep(0.01)
    cache_db.put("q4", schema, None, prediction)
    
    # q1 (the oldest) should be evicted
    assert cache_db.get("q1", schema, None) is None
    assert cache_db.get("q2", schema, None) is not None
    assert cache_db.get("q3", schema, None) is not None
    assert cache_db.get("q4", schema, None) is not None


def test_cache_concurrency(cache_db):
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    prediction = {
        "status": "completed",
        "query_ir": {"intent": "show"},
        "validation": {"is_valid": True},
    }
    
    errors = []
    
    def worker(num):
        try:
            for i in range(10):
                cache_db.put(f"worker_{num}_q_{i}", schema, None, prediction)
                cache_db.get(f"worker_{num}_q_{i}", schema, None)
        except Exception as exc:
            errors.append(exc)
            
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert len(errors) == 0, f"Concurrency errors: {errors}"
