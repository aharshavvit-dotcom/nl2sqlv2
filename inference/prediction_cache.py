"""Thread-safe prediction cache for NL-to-SQL inference.

Caches model predictions based on deterministic hashes of inputs to avoid
redundant inference calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PredictionCache:
    """A thread-safe, persistent LRU-evicting cache for NL-to-SQL predictions."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        max_entries: int = 1000,
    ) -> None:
        if cache_path is None:
            # Locate relative to project root
            root = Path(__file__).resolve().parents[1]
            self.cache_path = root / "artifacts" / "prediction_cache.json"
        else:
            self.cache_path = Path(cache_path)
        
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = {}
        self.lru_order: list[str] = []
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        with self.lock:
            if not self.cache_path.exists():
                self.cache = {}
                self.lru_order = []
                return
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.cache = data.get("cache", {})
                self.lru_order = data.get("lru_order", [])
                # Sync LRU list with cache keys to avoid discrepancy
                self.lru_order = [k for k in self.lru_order if k in self.cache]
                for k in self.cache:
                    if k not in self.lru_order:
                        self.lru_order.append(k)
            except Exception as exc:
                logger.warning("Failed to load prediction cache: %s. Starting fresh.", exc)
                self.cache = {}
                self.lru_order = []

    def _save_cache(self) -> None:
        """Write cache state to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.cache_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps({"cache": self.cache, "lru_order": self.lru_order}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            # Windows safe replace
            if self.cache_path.exists():
                self.cache_path.unlink()
            temp_path.rename(self.cache_path)
        except Exception as exc:
            logger.error("Failed to save prediction cache: %s", exc)

    def generate_hash_key(
        self,
        question: str,
        schema: dict[str, Any] | None,
        model_checkpoint_path: str | Path | None,
        routing_policy: dict[str, Any] | None = None,
    ) -> str:
        """Generate a deterministic SHA256 key from inputs."""
        # Normalize schema and routing parameters
        normalized_schema = json.dumps(schema or {}, sort_keys=True)
        normalized_routing = json.dumps(routing_policy or {}, sort_keys=True)
        model_path_str = str(Path(model_checkpoint_path).resolve()) if model_checkpoint_path else ""

        payload = {
            "question": str(question).strip(),
            "schema": normalized_schema,
            "model_path": model_path_str,
            "routing": normalized_routing,
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
        """Retrieve prediction from cache, updating LRU order."""
        if bypass_cache:
            return None

        key = self.generate_hash_key(question, schema, model_checkpoint_path, routing_policy)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            # Update LRU ordering
            if key in self.lru_order:
                self.lru_order.remove(key)
            self.lru_order.append(key)
            return self.cache[key]["prediction"]

    def put(
        self,
        question: str,
        schema: dict[str, Any] | None,
        model_checkpoint_path: str | Path | None,
        prediction: dict[str, Any],
        routing_policy: dict[str, Any] | None = None,
    ) -> None:
        """Store a prediction, evicting oldest entry if full."""
        key = self.generate_hash_key(question, schema, model_checkpoint_path, routing_policy)

        with self.lock:
            # Overwrite or store
            self.cache[key] = {
                "question": question,
                "prediction": prediction,
            }

            if key in self.lru_order:
                self.lru_order.remove(key)
            self.lru_order.append(key)

            # Enforce cache capacity limit
            while len(self.lru_order) > self.max_entries:
                oldest_key = self.lru_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]

            self._save_cache()

    def clear(self) -> None:
        """Clear cache contents."""
        with self.lock:
            self.cache = {}
            self.lru_order = []
            if self.cache_path.exists():
                try:
                    self.cache_path.unlink()
                except Exception as exc:
                    logger.error("Failed to delete cache file: %s", exc)
