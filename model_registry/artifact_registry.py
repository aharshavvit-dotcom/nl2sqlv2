from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import ModelManifest


class ArtifactRegistry:
    def __init__(self, registry_root: str | Path = "artifacts"):
        self.registry_root = Path(registry_root)

    def register_model(self, artifact_dir: str | Path, manifest: dict[str, Any] | ModelManifest) -> None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump() if isinstance(manifest, ModelManifest) else ModelManifest.model_validate(manifest).model_dump()
        (artifact_path / "model_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_models(self) -> list[dict[str, Any]]:
        if not self.registry_root.exists():
            return []
        rows = []
        for path in self.registry_root.glob("**/model_manifest.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows.append({**payload, "artifact_dir": str(path.parent)})
        return sorted(rows, key=lambda row: (str(row.get("model_name")), str(row.get("model_version"))))

    def get_latest_passing_model(self, model_name: str) -> dict[str, Any] | None:
        candidates = [
            row
            for row in self.list_models()
            if row.get("model_name") == model_name and (row.get("quality_gate") or {}).get("passed") is True
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: (str(row.get("created_at")), str(row.get("model_version"))), reverse=True)[0]
