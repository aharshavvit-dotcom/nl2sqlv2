"""Thread-safe, privacy-compliant SQLite prediction cache for NL-to-SQL inference.

Caches model predictions based on secure, dialect-aware, and tenant-isolated
hash keys to avoid redundant inference calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class PredictionCache:
    """A thread-safe, persistent SQLite LRU-evicting cache for NL-to-SQL predictions."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        max_entries: int = 1000,
        ttl_days: int = 7,
    ) -> None:
        if cache_path is None:
            root = Path(__file__).resolve().parents[1]
            # Use SQLite db path
            self.cache_db_path = root / "artifacts" / "prediction_cache.db"
        else:
            self.cache_db_path = Path(cache_path)
            
        self.max_entries = max_entries
        self.ttl_seconds = ttl_days * 86400
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection with Wal mode and synchronous defaults."""
        conn = sqlite3.connect(str(self.cache_db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create the cache table and indices if they do not exist."""
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    hash_key TEXT PRIMARY KEY,
                    normalized_question_hash TEXT,
                    schema_fingerprint TEXT,
                    bundle_id TEXT,
                    neural_checkpoint_hash TEXT,
                    retrieval_manifest_hash TEXT,
                    routing_policy_hash TEXT,
                    dialect TEXT,
                    tenant_id TEXT,
                    prediction_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_cache_last_accessed ON prediction_cache(last_accessed_at)")
            conn.commit()

    def generate_hash_key(
        self,
        question: str,
        schema: dict[str, Any] | None,
        model_checkpoint_path: str | Path | None,
        routing_policy: dict[str, Any] | None = None,
        cache_schema_version: str = "1.0",
    ) -> str:
        """Generate a deterministic SHA256 key from inputs."""
        # 1. Normalize and hash question (never cache raw questions)
        normalized_q = re.sub(r"\s+", " ", question.strip().lower())
        q_hash = hashlib.sha256(normalized_q.encode("utf-8")).hexdigest()

        # 2. Extract schema fingerprint and tenant
        schema_dict = schema or {}
        schema_fingerprint = schema_dict.get("schema_fingerprint", "")
        if not schema_fingerprint:
            # Fallback schema hash
            serialized_schema = json.dumps(schema_dict, sort_keys=True)
            schema_fingerprint = hashlib.sha256(serialized_schema.encode("utf-8")).hexdigest()
            
        tenant_id = schema_dict.get("tenant_id", "default")
        dialect = schema_dict.get("dialect", "sqlite")

        # 3. Resolve bundle and checkpoint identities from directory
        bundle_id = ""
        neural_checkpoint_hash = ""
        retrieval_manifest_hash = ""
        routing_policy_hash = ""

        if model_checkpoint_path:
            manifest_path = Path(model_checkpoint_path).parent / "bundle_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    bundle_id = manifest.get("bundle_id", "")
                    neural_checkpoint_hash = manifest.get("training_config_hash", "")
                    retrieval_manifest_hash = manifest.get("artifacts", {}).get("retrieval_manifest", "")
                except Exception:
                    pass

        # 4. Hash routing policy
        routing_policy_hash = hashlib.sha256(
            json.dumps(routing_policy or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        payload = {
            "cache_schema_version": cache_schema_version,
            "bundle_id": bundle_id,
            "neural_checkpoint_hash": neural_checkpoint_hash,
            "retrieval_manifest_hash": retrieval_manifest_hash,
            "routing_policy_hash": routing_policy_hash,
            "schema_fingerprint": schema_fingerprint,
            "dialect": dialect,
            "question_hash": q_hash,
            "tenant_id": tenant_id,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(
        self,
        question: str,
        schema: dict[str, Any] | None,
        model_checkpoint_path: str | Path | None,
        routing_policy: dict[str, Any] | None = None,
        bypass_cache: bool = False,
    ) -> dict[str, Any] | None:
        """Retrieve prediction from cache, updating LRU order approximately."""
        if bypass_cache:
            return None

        key = self.generate_hash_key(question, schema, model_checkpoint_path, routing_policy)
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT prediction_json, last_accessed_at FROM prediction_cache WHERE hash_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if not row:
                return None
                
            prediction_json, last_accessed_str = row
            prediction = json.loads(prediction_json)

            # Approximate LRU: update access time only if older than 60s to limit write locks
            try:
                # SQLite TIMESTAMP handles text datetimes
                last_accessed = datetime.strptime(last_accessed_str[:19], "%Y-%m-%d %H:%M:%S")
                if datetime.utcnow() - last_accessed > timedelta(seconds=60):
                    cursor.execute(
                        "UPDATE prediction_cache SET last_accessed_at = CURRENT_TIMESTAMP WHERE hash_key = ?",
                        (key,),
                    )
                    conn.commit()
            except Exception:
                pass

            return prediction
        except Exception as exc:
            logger.warning("Cache fetch error: %s", exc)
            return None
        finally:
            conn.close()

    def put(
        self,
        question: str,
        schema: dict[str, Any] | None,
        model_checkpoint_path: str | Path | None,
        prediction: dict[str, Any],
        routing_policy: dict[str, Any] | None = None,
    ) -> None:
        """Store a prediction, enforcing TTL and LRU evictions atomically."""
        # Rule 16: Cache only validated, successful, non-failed predictions
        status = prediction.get("status")
        sql_val = prediction.get("validation") or {}
        query_ir = prediction.get("query_ir") or {}
        
        is_eligible = (
            status == "completed"
            and sql_val.get("is_valid", sql_val.get("ok", False))
            and bool(query_ir)
        )
        if not is_eligible:
            return

        key = self.generate_hash_key(question, schema, model_checkpoint_path, routing_policy)
        normalized_q = re.sub(r"\s+", " ", question.strip().lower())
        q_hash = hashlib.sha256(normalized_q.encode("utf-8")).hexdigest()
        
        schema_dict = schema or {}
        schema_fingerprint = schema_dict.get("schema_fingerprint", "")
        if not schema_fingerprint:
            schema_fingerprint = hashlib.sha256(json.dumps(schema_dict, sort_keys=True).encode("utf-8")).hexdigest()
            
        tenant_id = schema_dict.get("tenant_id", "default")
        dialect = schema_dict.get("dialect", "sqlite")

        # Resolve bundle metadata
        bundle_id = ""
        neural_checkpoint_hash = ""
        retrieval_manifest_hash = ""
        if model_checkpoint_path:
            manifest_path = Path(model_checkpoint_path).parent / "bundle_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    bundle_id = manifest.get("bundle_id", "")
                    neural_checkpoint_hash = manifest.get("training_config_hash", "")
                    retrieval_manifest_hash = manifest.get("artifacts", {}).get("retrieval_manifest", "")
                except Exception:
                    pass

        routing_policy_hash = hashlib.sha256(
            json.dumps(routing_policy or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        # Sanitize prediction output (Rule 10: do not store raw question or SQL literals)
        prediction_copy = dict(prediction)
        if "question" in prediction_copy:
            del prediction_copy["question"]
            
        # Optional: remove sensitive literal values from query_ir filters
        if "query_ir" in prediction_copy:
            ir_copy = dict(prediction_copy["query_ir"])
            if "filters" in ir_copy:
                filters_copy = []
                for filt in ir_copy["filters"]:
                    f_copy = dict(filt)
                    if "value" in f_copy:
                        f_copy["value"] = "[REDACTED]"
                    filters_copy.append(f_copy)
                ir_copy["filters"] = filters_copy
            prediction_copy["query_ir"] = ir_copy

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            
            # Insert or replace prediction
            conn.execute(
                """
                INSERT OR REPLACE INTO prediction_cache (
                    hash_key, normalized_question_hash, schema_fingerprint,
                    bundle_id, neural_checkpoint_hash, retrieval_manifest_hash,
                    routing_policy_hash, dialect, tenant_id, prediction_json,
                    last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    key, q_hash, schema_fingerprint, bundle_id, neural_checkpoint_hash,
                    retrieval_manifest_hash, routing_policy_hash, dialect, tenant_id,
                    json.dumps(prediction_copy),
                ),
            )

            # Evict expired entries (TTL)
            conn.execute(
                "DELETE FROM prediction_cache WHERE created_at <= datetime('now', ?)",
                (f"-{self.ttl_seconds} seconds",),
            )

            # Enforce max entry limit using subquery eviction
            conn.execute(
                """
                DELETE FROM prediction_cache
                WHERE hash_key IN (
                    SELECT hash_key
                    FROM prediction_cache
                    ORDER BY last_accessed_at ASC
                    LIMIT MAX(
                        0,
                        (SELECT COUNT(*) FROM prediction_cache) - ?
                    )
                )
                """,
                (self.max_entries,),
            )
            
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to insert prediction in cache: %s", exc)
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear cache database."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM prediction_cache")
            conn.commit()
        except Exception as exc:
            logger.error("Failed to clear SQLite cache: %s", exc)
        finally:
            conn.close()
