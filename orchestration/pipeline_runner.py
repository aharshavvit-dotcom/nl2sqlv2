from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline_config import PipelineConfig
from .pipeline_reporter import PipelineReporter
from .pipeline_state import PipelineState
from .step_runner import StepRunner


class PipelineRunner:
    def __init__(self, state_path: str | Path = "artifacts/pipeline/pipeline_state.json"):
        self.state = PipelineState(state_path)
        self.steps = StepRunner()

    def run(self, config_path: str, start_at: str | None = None, stop_after: str | None = None) -> dict[str, Any]:
        config = PipelineConfig.load(config_path)
        self.state.load()
        selected_steps = _slice_steps(config.steps, start_at, stop_after)
        results = []
        status = "completed"
        for step in selected_steps:
            self.state.update_step(step, "running")
            try:
                result = self.steps.run_step(step, config)
                step_status = result.get("status", "completed")
                if step_status == "skipped":
                    self.state.update_step(step, "skipped", result)
                else:
                    self.state.update_step(step, "completed", result)
                results.append({"step": step, **result})
            except Exception as exc:
                status = "failed"
                failure = {"step": step, "status": "failed", "error": str(exc)}
                self.state.update_step(step, "failed", failure)
                results.append(failure)
                break
        report = {"pipeline_name": config.pipeline_name, "status": status, "steps": results}
        PipelineReporter().write("artifacts/pipeline", report)
        return report


def _slice_steps(steps: list[str], start_at: str | None, stop_after: str | None) -> list[str]:
    start = steps.index(start_at) if start_at in steps else 0
    end = steps.index(stop_after) + 1 if stop_after in steps else len(steps)
    return steps[start:end]
