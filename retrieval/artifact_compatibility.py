"""Version metadata and fail-closed checks for serialized sklearn artifacts.

This module exists because both legacy TF-IDF artifacts and the RAG indexes
contain sklearn objects and must follow the same compatibility policy.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn


METADATA_FILE = "sklearn_artifact_metadata.json"


def build_sklearn_metadata(
    *,
    artifact_types: list[str],
    source_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "artifact_created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_type": artifact_types[0] if len(artifact_types) == 1 else "sklearn_artifact_bundle",
        "artifact_types": artifact_types,
        "source_dataset_hash": _file_hash(source_path),
        "config_hash": hashlib.sha256(
            json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }


def write_sklearn_metadata(directory: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(directory) / METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_sklearn_metadata(
    directory: str | Path,
    *,
    mode: str = "runtime",
) -> dict[str, Any]:
    """Validate metadata before unpickling; training mode signals rebuild."""
    path = Path(directory) / METADATA_FILE
    if not path.exists():
        message = f"Missing sklearn artifact metadata: {path}"
        if mode == "training":
            return {"compatible": False, "rebuild_required": True, "reason": message}
        raise RuntimeError(message + ". Rebuild the retrieval artifacts before loading them.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    saved = str(metadata.get("sklearn_version") or "")
    current = str(sklearn.__version__)
    if saved != current:
        message = f"Incompatible sklearn artifact version: saved={saved or 'unknown'}, runtime={current}"
        if mode == "training":
            return {"compatible": False, "rebuild_required": True, "reason": message, "metadata": metadata}
        raise RuntimeError(message + ". Production/runtime loading fails closed; rebuild the artifact.")
    return {"compatible": True, "rebuild_required": False, "metadata": metadata}


def _file_hash(path: str | Path | None) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
