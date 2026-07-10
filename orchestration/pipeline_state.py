from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineState:
    def __init__(
        self,
        path: str | Path = "artifacts/pipeline/pipeline_state.json",
        *,
        pipeline_run_id: str = "",
        effective_config_hash: str = "",
    ):
        self.path = Path(path)
        self.pipeline_run_id = pipeline_run_id
        self.effective_config_hash = effective_config_hash
        self.state: dict[str, Any] = {
            "state_schema_version": "1.0",
            "pipeline_run_id": pipeline_run_id,
            "effective_config_hash": effective_config_hash,
            "steps": {},
            "last_completed_step": None,
            "failed_step": None,
        }

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        self.state.setdefault("state_schema_version", "1.0")
        if self.pipeline_run_id and not self.state.get("pipeline_run_id"):
            self.state["pipeline_run_id"] = self.pipeline_run_id
        if self.effective_config_hash and not self.state.get("effective_config_hash"):
            self.state["effective_config_hash"] = self.effective_config_hash
        return self.state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def update_step(self, step: str, status: str, details: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        existing = self.state.setdefault("steps", {}).get(step, {})
        entry: dict[str, Any] = {
            "step": step,
            "status": status,
            "started_at": existing.get("started_at") or now,
            "ended_at": now if status in {"completed", "failed", "skipped"} else None,
            "inputs": (details or {}).get("inputs", existing.get("inputs", [])),
            "outputs": (details or {}).get("outputs", existing.get("outputs", [])),
            "error": (details or {}).get("error"),
            "skip_reason": (details or {}).get("skip_reason") or (details or {}).get("reason"),
            "pipeline_run_id": (details or {}).get("pipeline_run_id")
            or self.state.get("pipeline_run_id")
            or self.pipeline_run_id,
            "effective_config_hash": (details or {}).get("effective_config_hash")
            or self.state.get("effective_config_hash")
            or self.effective_config_hash,
            "step_contract_version": (details or {}).get("step_contract_version", "1.0"),
            "details": details or {},
        }
        self.state["steps"][step] = entry
        if status == "completed":
            self.state["last_completed_step"] = step
            self.state["failed_step"] = None
        if status == "failed":
            self.state["failed_step"] = step
        self.save()

    def get_step_status(self, step: str) -> str | None:
        """Return the status of a specific step, or None if not recorded."""
        entry = self.state.get("steps", {}).get(step)
        return entry.get("status") if entry else None

    def can_reuse_step(self, step: str, *, pipeline_run_id: str, effective_config_hash: str) -> bool:
        entry = self.state.get("steps", {}).get(step)
        if not entry or entry.get("status") != "completed":
            return False
        return (
            str(entry.get("pipeline_run_id") or self.state.get("pipeline_run_id") or "") == str(pipeline_run_id)
            and str(entry.get("effective_config_hash") or self.state.get("effective_config_hash") or "")
            == str(effective_config_hash)
            and str(entry.get("step_contract_version") or "") == "1.0"
        )

    def all_steps_summary(self) -> list[dict[str, Any]]:
        """Return a summary list of all recorded step states."""
        results = []
        for step_name, entry in self.state.get("steps", {}).items():
            results.append({
                "step": step_name,
                "status": entry.get("status"),
                "started_at": entry.get("started_at"),
                "ended_at": entry.get("ended_at"),
                "error": entry.get("error"),
                "skip_reason": entry.get("skip_reason"),
            })
        return results

    def has_silently_skipped_steps(self) -> bool:
        """Return True if any step was skipped without a reason."""
        for entry in self.state.get("steps", {}).values():
            if entry.get("status") == "skipped" and not entry.get("skip_reason"):
                return True
        return False
