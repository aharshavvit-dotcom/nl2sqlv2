"""Training cache for tokenized / linearized examples.

Speeds up repeated training runs by caching intermediate representations.
Cache key is a SHA-256 of the content, stored under ``artifacts/cache/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class TrainingCache:
    """File-backed cache keyed by content hash.

    Parameters
    ----------
    cache_dir:
        Directory for cache files.  Created on first write.
    """

    def __init__(self, cache_dir: str | Path = "artifacts/cache") -> None:
        self.cache_dir = Path(cache_dir)

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None``."""
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8",
        )

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def clear(self) -> int:
        """Remove all cached files.  Returns number of files removed."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for item in self.cache_dir.glob("*.json"):
            item.unlink()
            count += 1
        return count

    def content_key(self, content: str) -> str:
        """Compute a cache key from raw content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]

    # ── private ───────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
