"""Create a pre-hardening baseline manifest.

The manifest is deliberately evidence-oriented: if a file, dependency, or
report is missing, the output records that fact instead of inventing a value.
It does not modify trained artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_files_under(relative: str) -> dict[str, dict[str, Any]]:
    base = ROOT / relative
    result: dict[str, dict[str, Any]] = {}
    if not base.exists():
        return result
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        rel = path.relative_to(ROOT).as_posix()
        try:
            result[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        except OSError as exc:
            result[rel] = {"error": str(exc)}
    return result


def read_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"available": False, "path": relative}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "path": relative, "error": str(exc)}
    if isinstance(payload, dict):
        return {"available": True, "path": relative, "payload": payload}
    return {"available": True, "path": relative, "payload_type": type(payload).__name__}


def read_json_evidence(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"available": False, "path": relative}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "path": relative, "error": str(exc)}
    evidence: dict[str, Any] = {
        "available": True,
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "payload_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        for key in (
            "summary",
            "metrics",
            "passed",
            "strict_passed",
            "status",
            "overall_status",
            "pipeline_run_id",
            "bundle_id",
            "checkpoint",
        ):
            if key in payload:
                evidence[key] = payload[key]
        if "results" in payload and isinstance(payload["results"], list):
            evidence["result_count"] = len(payload["results"])
    return evidence


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for module_name in ("yaml", "numpy", "sklearn", "joblib", "torch", "sqlglot", "pydantic"):
        try:
            module = __import__(module_name)
        except Exception as exc:  # pragma: no cover - environment evidence
            versions[module_name] = f"unavailable: {exc}"
            continue
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def canonical_state_dict_hash(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"available": False, "path": relative}
    try:
        import torch
    except Exception as exc:
        return {"available": False, "path": relative, "error": f"torch unavailable: {exc}"}
    try:
        obj = torch.load(path, map_location="cpu")
        state_dict = obj.get("state_dict") if isinstance(obj, dict) else obj
        if not isinstance(state_dict, dict):
            return {"available": False, "path": relative, "error": "object is not a state dict"}
        hasher = hashlib.sha256()
        tensor_keys = 0
        for key in sorted(state_dict):
            value = state_dict[key]
            hasher.update(str(key).encode("utf-8"))
            if hasattr(value, "detach"):
                tensor = value.detach().cpu().contiguous()
                hasher.update(str(tensor.dtype).encode("utf-8"))
                hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
                hasher.update(tensor.numpy().tobytes())
                tensor_keys += 1
            else:
                hasher.update(repr(value).encode("utf-8"))
        return {
            "available": True,
            "path": relative,
            "state_dict_sha256": hasher.hexdigest(),
            "tensor_keys": tensor_keys,
        }
    except Exception as exc:
        return {"available": False, "path": relative, "error": str(exc)}


def main() -> int:
    config_path = ROOT / "configs" / "training.yaml"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    manifest = {
        "baseline_schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "purpose": "pre-implementation baseline manifest for semantic hardening",
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "status_short": git_output("status", "--short"),
        },
        "dependency_versions": dependency_versions(),
        "checkpoint_file_hashes": hash_files_under("artifacts/work/neural_ir"),
        "canonical_state_dict_hashes": {
            "best_model.pt": canonical_state_dict_hash("artifacts/work/neural_ir/best_model.pt"),
            "model.pt": canonical_state_dict_hash("artifacts/work/neural_ir/model.pt"),
        },
        "retrieval_artifact_hashes": hash_files_under("artifacts/work/retrieval_ir"),
        "effective_training_config_hash": (
            hashlib.sha256(config_text.encode("utf-8")).hexdigest() if config_text else None
        ),
        "evaluation_dataset_hashes": hash_files_under("data/processed"),
        "current_semantic_metrics": {
            "generic_model_evaluation": read_json_evidence("artifacts/evaluation/generic_model_evaluation_report.json"),
            "dev_existing_checkpoint": read_json_evidence(
                "artifacts/evaluation/dev_existing_checkpoint/generic_model_evaluation_report.json"
            ),
            "quality_gate": read_json_evidence("artifacts/evaluation/model_quality_gate_report.json"),
        },
        "current_route_metrics": read_json_evidence("artifacts/evaluation/route_diagnostics_report.json"),
        "current_application_configuration": {
            "training_yaml_path": "configs/training.yaml",
            "training_yaml_hash": hashlib.sha256(config_text.encode("utf-8")).hexdigest()
            if config_text
            else None,
            "model_bundle_current": read_json("artifacts/model_bundle/current/bundle_manifest.json"),
            "model_bundle_latest_candidate": read_json("artifacts/model_bundle/latest_candidate.json"),
        },
        "preserved_artifacts": {
            "neural_ir": "artifacts/work/neural_ir",
            "retrieval_ir": "artifacts/work/retrieval_ir",
            "backup_production_training_20260707": "artifacts/backup/production_training_20260707",
            "dev_existing_checkpoint": "artifacts/evaluation/dev_existing_checkpoint",
        },
        "notes": [
            "Missing files are recorded as unavailable evidence, not silently accepted.",
            "No preserved model artifact is modified by this manifest generation.",
        ],
    }
    output = ROOT / "artifacts" / "audit" / "baseline_manifest_pre_hardening.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        os.chmod(output, stat.S_IWRITE | stat.S_IREAD)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(output, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print(output)
    best_hash = manifest["checkpoint_file_hashes"].get(
        "artifacts/work/neural_ir/best_model.pt", {}
    ).get("sha256")
    print(f"best_model.pt sha256: {best_hash or 'unavailable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
