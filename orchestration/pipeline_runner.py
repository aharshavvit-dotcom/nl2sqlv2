from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract_validator import ContractValidator
from .pipeline_config import PipelineConfig
from .pipeline_reporter import PipelineReporter
from .pipeline_state import PipelineState
from .step_contract import StepContract
from .step_runner import StepRunner


class PipelineRunner:
    def __init__(self, state_path: str | Path = "artifacts/pipeline/pipeline_state.json"):
        self.state = PipelineState(state_path)
        self.steps = StepRunner()
        self.contract_validator = ContractValidator()

    def run(
        self,
        config_path: str,
        start_at: str | None = None,
        stop_after: str | None = None,
        resume: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        config = PipelineConfig.load(config_path)
        self.state.load()
        selected_steps = _slice_steps(config.steps, start_at, stop_after)

        # Resume: skip already-completed steps
        if resume and not force:
            selected_steps = [
                step for step in selected_steps
                if self.state.get_step_status(step) != "completed"
            ]

        results = []
        status = "completed"
        for step in selected_steps:
            contract = self.steps.get_contract(step, config)

            # Check if step is skippable
            if contract.can_skip:
                self.state.update_step(step, "skipped", {
                    "skip_reason": contract.skip_reason or "disabled in config",
                })
                results.append({"step": step, "status": "skipped", "reason": contract.skip_reason})
                continue

            # Validate inputs (fail-fast)
            if contract.inputs and not dry_run:
                input_check = self.contract_validator.validate_inputs(contract, _root())
                if not input_check["valid"] and contract.required:
                    error_msg = f"Missing required inputs for {step}: {input_check['missing']}"
                    if not force:
                        status = "failed"
                        failure = {"step": step, "status": "failed", "error": error_msg}
                        self.state.update_step(step, "failed", failure)
                        results.append(failure)
                        break

            if dry_run:
                results.append({"step": step, "status": "dry_run", "contract": {
                    "inputs": contract.inputs, "outputs": contract.outputs,
                    "required": contract.required, "can_skip": contract.can_skip,
                }})
                continue

            self.state.update_step(step, "running")
            try:
                result = self.steps.run_step(step, config)
                step_status = result.get("status", "completed")
                if step_status == "skipped":
                    self.state.update_step(step, "skipped", result)
                else:
                    self.state.update_step(step, "completed", result)
                results.append({"step": step, **result})

                # Validate outputs (fail-fast)
                if contract.outputs and step_status != "skipped":
                    output_check = self.contract_validator.validate_outputs(contract, _root())
                    if not output_check["valid"] and contract.required:
                        if not force:
                            error_msg = f"Missing required outputs from {step}: {output_check['missing']}"
                            status = "failed"
                            failure = {"step": step, "status": "output_validation_failed", "error": error_msg}
                            self.state.update_step(step, "failed", failure)
                            results.append(failure)
                            break

            except Exception as exc:
                status = "failed"
                failure = {"step": step, "status": "failed", "error": str(exc)}
                self.state.update_step(step, "failed", failure)
                results.append(failure)
                break

        report = {"pipeline_name": config.pipeline_name, "status": status, "steps": results}
        if not dry_run:
            PipelineReporter().write("artifacts/pipeline", report)
        return report


def _slice_steps(steps: list[str], start_at: str | None, stop_after: str | None) -> list[str]:
    start = steps.index(start_at) if start_at in steps else 0
    end = steps.index(stop_after) + 1 if stop_after in steps else len(steps)
    return steps[start:end]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]
