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

try:
    import numpy
    _NUMPY_VERSION = numpy.__version__
except ImportError:
    _NUMPY_VERSION = "unknown"

try:
    import joblib
    _JOBLIB_VERSION = joblib.__version__
except ImportError:
    _JOBLIB_VERSION = "unknown"


METADATA_FILE = "sklearn_artifact_metadata.json"


def build_sklearn_metadata(
    *,
    artifact_types: list[str],
    source_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    pickle_filenames: list[str] | None = None,
) -> dict[str, Any]:
    effective_config_hash = hashlib.sha256(
        json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    metadata: dict[str, Any] = {
        "artifact_schema_version": "1.0",
        "sklearn_version": sklearn.__version__,
        "numpy_version": _NUMPY_VERSION,
        "joblib_version": _JOBLIB_VERSION,
        "python_version": platform.python_version(),
        "artifact_created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_type": artifact_types[0] if len(artifact_types) == 1 else "sklearn_artifact_bundle",
        "artifact_types": artifact_types,
        "source_dataset_hash": _file_hash(source_path),
        "effective_config_hash": effective_config_hash,
        "config_hash": effective_config_hash,
    }
    # Add SHA256 checksums of pickle files
    if artifact_dir and pickle_filenames:
        file_checksums: dict[str, str] = {}
        for filename in pickle_filenames:
            file_path = Path(artifact_dir) / filename
            if file_path.exists():
                file_checksums[filename] = _file_hash(file_path)
        metadata["files"] = file_checksums
    return metadata


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


def validate_file_checksums(
    directory: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate SHA256 checksums of pickle files against metadata.

    Returns:
        Dict with 'valid', 'mismatched_files', and 'missing_files'.
    """
    dir_path = Path(directory)
    if metadata is None:
        meta_path = dir_path / METADATA_FILE
        if not meta_path.exists():
            return {"valid": False, "reason": "metadata_missing", "mismatched_files": [], "missing_files": []}
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    recorded_checksums = metadata.get("files") or {}
    if not recorded_checksums:
        return {"valid": True, "reason": "no_checksums_recorded", "mismatched_files": [], "missing_files": []}

    mismatched: list[str] = []
    missing: list[str] = []
    for filename, expected_hash in recorded_checksums.items():
        file_path = dir_path / filename
        if not file_path.exists():
            missing.append(filename)
            continue
        actual_hash = _file_hash(file_path)
        if actual_hash != expected_hash:
            mismatched.append(filename)

    return {
        "valid": len(mismatched) == 0 and len(missing) == 0,
        "mismatched_files": mismatched,
        "missing_files": missing,
    }


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
