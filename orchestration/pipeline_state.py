from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PipelineState:
    def __init__(self, path: str | Path = "artifacts/pipeline/pipeline_state.json"):
        self.path = Path(path)
        self.state: dict[str, Any] = {"steps": {}, "last_completed_step": None, "failed_step": None}

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        return self.state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def update_step(self, step: str, status: str, details: dict[str, Any] | None = None) -> None:
        self.state.setdefault("steps", {})[step] = {"status": status, "details": details or {}}
        if status == "completed":
            self.state["last_completed_step"] = step
            self.state["failed_step"] = None
        if status == "failed":
            self.state["failed_step"] = step
        self.save()
