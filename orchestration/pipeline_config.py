from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_STEPS = [
    "audit_execution_pipeline",
    "audit_self_training",
    "build_generic_ir_corpus",
    "build_retrieval_rag_index",
    "train_neural_ir_model",
    "evaluate_against_gold",
    "mine_validation_errors",
    "build_corrections_from_gold",
    "train_ranking_from_gold",
    "run_self_improvement_loop",
    "run_execution_aware_evaluation",
    "evaluate_generic_models",
    "select_best_model",
    "promote_model_if_better",
    "build_semantic_profile",
    "generate_connected_db_regressions",
    "run_connected_db_regressions",
    "run_app_smoke_check",
]


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

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            pipeline_name=payload.get("pipeline_name", Path(path).stem),
            seed=int(payload.get("seed", 42)),
            datasets=payload.get("datasets") or {},
            training=payload.get("training") or {},
            artifacts=payload.get("artifacts") or {},
            steps=payload.get("steps") or list(DEFAULT_STEPS),
            smoke=bool(payload.get("smoke", False)),
            skip_heavy_steps=bool(payload.get("skip_heavy_steps", False)),
        )
