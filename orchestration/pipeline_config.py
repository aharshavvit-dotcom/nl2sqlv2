from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def build_pipeline_steps(config: dict[str, Any]) -> list[str]:
    """Return the sole canonical ordered training-step registry.

    Both public training entry points call this function. Optional components
    remain represented by contracts that explicitly report disabled steps.
    """
    retrieval = config.get("retrieval") or {}
    neural = config.get("neural") or {}
    self_training = config.get("self_training") or {}
    evaluation = config.get("evaluation") or {}
    bundle = config.get("bundle") or {}

    steps = ["verify_datasets"]
    if retrieval.get("enabled", True) or neural.get("enabled", True):
        steps.append("build_generic_ir_corpus")
    if retrieval.get("enabled", True):
        steps.append("build_retrieval_rag_index")
    if neural.get("enabled", True):
        steps.extend(["build_hard_negative_corpus", "train_neural_ir"])
    if self_training.get("enabled", False) or evaluation.get("enabled", True):
        steps.append("evaluate_against_gold")
    steps.extend(["mine_validation_errors", "build_corrections_from_gold", "train_adaptive_ranker"])
    if self_training.get("enabled", False):
        steps.append("run_self_improvement_loop")
    if evaluation.get("enabled", True):
        if evaluation.get("run_execution_aware", False):
            steps.append("run_execution_aware_evaluation")
        steps.append("evaluate_generic_models")
    steps.append("run_quality_gate")
    if bundle.get("build", True):
        steps.append("build_model_bundle")
        if bundle.get("validate", True):
            steps.append("validate_model_bundle")
        if bundle.get("promote_if_quality_gate_passes", False):
            steps.append("promote_model_bundle")
    if config.get("smoke", False):
        steps.append("run_app_smoke_check")
    return steps


DEFAULT_STEPS = build_pipeline_steps({})


@dataclass
class PipelineConfig:
    pipeline_name: str
    seed: int = 42
    datasets: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps: list[str] = field(default_factory=lambda: list(DEFAULT_STEPS))
    smoke: bool = False
    skip_heavy_steps: bool = False
    integrated_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        integrated_config = payload.get("_integrated_config") or {}
        training = payload.get("training") or {}
        if integrated_config:
            training = {**training, "_integrated_config": integrated_config}
        return cls(
            pipeline_name=payload.get("pipeline_name", Path(path).stem),
            seed=int(payload.get("seed", 42)),
            datasets=payload.get("datasets") or {},
            training=training,
            artifacts=payload.get("artifacts") or {},
            steps=payload.get("steps") or build_pipeline_steps(integrated_config or payload),
            smoke=bool(payload.get("smoke", False)),
            skip_heavy_steps=bool(payload.get("skip_heavy_steps", False)),
            integrated_config=integrated_config,
        )
