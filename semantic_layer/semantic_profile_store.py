from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SemanticProfileStore:
    def __init__(self, root: str | Path = "artifacts/semantic_profiles"):
        self.root = Path(root)

    def save(self, schema_fingerprint: str, profile: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {**profile, "schema_fingerprint": schema_fingerprint}
        (self.root / f"{schema_fingerprint}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, schema_fingerprint: str) -> dict[str, Any] | None:
        path = self.root / f"{schema_fingerprint}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_profiles(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        rows = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "schema_fingerprint": payload.get("schema_fingerprint") or path.stem,
                    "database": payload.get("database"),
                    "schema_name": payload.get("schema_name"),
                    "table_count": len(payload.get("tables") or {}),
                    "path": str(path),
                }
            )
        return rows
