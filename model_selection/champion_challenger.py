from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class ChampionChallengerRegistry:
    def __init__(self, path: str | Path = "artifacts/model_registry/champion_challenger_registry.json"):
        self.path = Path(path)

    def get_current_champion(self, model_name: str) -> dict[str, Any] | None:
        registry = self._load()
        return (registry.get(model_name) or {}).get("champion")

    def register_challenger(self, model_name: str, artifact_dir: str, metrics: dict[str, Any]) -> dict[str, Any]:
        registry = self._load()
        entry = registry.setdefault(model_name, {"champion": None, "challengers": []})
        challenger = {
            "challenger_id": f"chg_{uuid4().hex}",
            "model_name": model_name,
            "artifact_dir": artifact_dir,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "challenger",
        }
        entry["challengers"].append(challenger)
        self._save(registry)
        return challenger

    def promote_challenger(self, model_name: str, challenger_id: str) -> dict[str, Any]:
        registry = self._load()
        entry = registry.setdefault(model_name, {"champion": None, "challengers": []})
        challenger = next((item for item in entry["challengers"] if item.get("challenger_id") == challenger_id), None)
        if challenger is None:
            raise ValueError(f"Unknown challenger_id: {challenger_id}")
        champion = {**challenger, "status": "champion", "promoted_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
        entry["champion"] = champion
        for item in entry["challengers"]:
            if item.get("challenger_id") == challenger_id:
                item["status"] = "promoted"
        self._save(registry)
        return champion

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
