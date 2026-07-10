from __future__ import annotations

import hashlib
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
        self.state_path = Path(state_path)
        self.state = PipelineState(self.state_path)
        self.output_dir = self.state_path.parent
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
        effective_config_hash = _sha256_text(Path(config_path).read_text(encoding="utf-8"))
        self.state.pipeline_run_id = config.pipeline_run_id
        self.state.effective_config_hash = effective_config_hash
        self.state.load()
        selected_steps = _slice_steps(config.steps, start_at, stop_after)

        # Resume: skip already-completed steps
        if resume and not force:
            selected_steps = [
                step for step in selected_steps
                if not self.state.can_reuse_step(
                    step,
                    pipeline_run_id=config.pipeline_run_id,
                    effective_config_hash=effective_config_hash,
                )
            ]

        results = []
        status = "completed"
        for step in selected_steps:
            try:
                contract = self.steps.get_contract(step, config)
            except Exception as exc:
                status = "failed"
                failure = {"step": step, "status": "failed", "error": str(exc)}
                self.state.update_step(step, "failed", _with_identity(failure, config, effective_config_hash))
                results.append(failure)
                break

            # Check if step is skippable
            if contract.can_skip:
                self.state.update_step(step, "skipped", _with_identity({
                    "skip_reason": contract.skip_reason or "disabled in config",
                }, config, effective_config_hash))
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
                        self.state.update_step(step, "failed", _with_identity(failure, config, effective_config_hash))
                        results.append(failure)
                        break

            if dry_run:
                results.append({"step": step, "status": "dry_run", "contract": {
                    "inputs": contract.inputs, "outputs": contract.outputs,
                    "required": contract.required, "can_skip": contract.can_skip,
                }})
                continue

            self.state.update_step(step, "running", _with_identity({}, config, effective_config_hash))
            try:
                if step == "build_model_bundle":
                    PipelineReporter().write(
                        self.output_dir,
                        {
                            "pipeline_name": config.pipeline_name,
                            "pipeline_run_id": config.pipeline_run_id,
                            "effective_config_hash": effective_config_hash,
                            "status": "running",
                            "steps": [*results, {"step": step, "status": "running"}],
                        },
                    )
                result = self.steps.run_step(step, config)
                step_status = result.get("status", "completed")
                if step_status == "skipped":
                    if contract.required:
                        status = "failed"
                        failure = {
                            "step": step,
                            "status": "failed",
                            "error": f"Required step {step} was skipped: {result.get('reason') or 'no reason'}",
                        }
                        self.state.update_step(step, "failed", _with_identity(failure, config, effective_config_hash))
                        results.append(failure)
                        break
                    self.state.update_step(step, "skipped", _with_identity(result, config, effective_config_hash))
                elif step_status == "failed":
                    status = "failed"
                    failure = {"step": step, **result}
                    self.state.update_step(step, "failed", _with_identity(failure, config, effective_config_hash))
                    results.append(failure)
                    break
                else:
                    self.state.update_step(step, "completed", _with_identity(result, config, effective_config_hash))
                results.append({"step": step, **result})

                # Validate outputs (fail-fast)
                if contract.outputs and step_status != "skipped":
                    output_check = self.contract_validator.validate_outputs(contract, _root())
                    if not output_check["valid"] and contract.required:
                        if not force:
                            error_msg = f"Missing required outputs from {step}: {output_check['missing']}"
                            status = "failed"
                            failure = {
                                "step": step,
                                "status": "failed",
                                "error": error_msg,
                                "validation_status": "output_validation_failed",
                            }
                            self.state.update_step(step, "failed", _with_identity(failure, config, effective_config_hash))
                            results.append(failure)
                            break

            except Exception as exc:
                status = "failed"
                failure = {"step": step, "status": "failed", "error": str(exc)}
                self.state.update_step(step, "failed", _with_identity(failure, config, effective_config_hash))
                results.append(failure)
                break

        report = {
            "pipeline_name": config.pipeline_name,
            "pipeline_run_id": config.pipeline_run_id,
            "effective_config_hash": effective_config_hash,
            "status": status,
            "steps": results,
        }
        if not dry_run:
            PipelineReporter().write(self.output_dir, report)
        return report


def _slice_steps(steps: list[str], start_at: str | None, stop_after: str | None) -> list[str]:
    if start_at is not None and start_at not in steps:
        raise ValueError(f"Unknown start step: {start_at}")
    if stop_after is not None and stop_after not in steps:
        raise ValueError(f"Unknown stop step: {stop_after}")
    start = steps.index(start_at) if start_at in steps else 0
    end = steps.index(stop_after) + 1 if stop_after in steps else len(steps)
    return steps[start:end]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _with_identity(payload: dict[str, Any], config: PipelineConfig, effective_config_hash: str) -> dict[str, Any]:
    return {
        **payload,
        "pipeline_run_id": config.pipeline_run_id,
        "effective_config_hash": effective_config_hash,
        "step_contract_version": "1.0",
    }
