from __future__ import annotations

import json
from pathlib import Path

from orchestration.pipeline_state import PipelineState
from training.train_model import main, parse_args


def test_pipeline_state_reuse_requires_same_run_and_config_hash(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline_state.json"
    state = PipelineState(state_path, pipeline_run_id="run-a", effective_config_hash="hash-a")
    state.load()
    state.update_step("step_a", "completed", {"pipeline_run_id": "run-a", "effective_config_hash": "hash-a"})

    loaded = PipelineState(state_path, pipeline_run_id="run-b", effective_config_hash="hash-a")
    loaded.load()

    assert loaded.can_reuse_step("step_a", pipeline_run_id="run-a", effective_config_hash="hash-a") is True
    assert loaded.can_reuse_step("step_a", pipeline_run_id="run-b", effective_config_hash="hash-a") is False
    assert loaded.can_reuse_step("step_a", pipeline_run_id="run-a", effective_config_hash="hash-b") is False


def test_train_model_resume_run_id_preserves_run_identity(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text("pipeline: {name: unit}\ndatasets: {names: []}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        "training.train_model.parse_args",
        lambda: type("Args", (), {
            "config": config_path,
            "start_at": None,
            "stop_after": None,
            "resume": False,
            "resume_run_id": "existing-run-123",
            "force": False,
            "dry_run": True,
        })(),
    )
    monkeypatch.setattr("training.train_model.validate_environment", lambda config: [])
    monkeypatch.setattr("training.train_model.verify_datasets", lambda config: True)

    def fake_run_pipeline(config, args):
        captured["run_id"] = config["_pipeline_run_id"]
        captured["resume"] = args.resume or bool(args.resume_run_id)
        return {"pipeline_name": "unit", "status": "completed", "steps": []}

    monkeypatch.setattr("training.train_model.run_pipeline", fake_run_pipeline)

    assert main() == 0
    assert captured == {"run_id": "existing-run-123", "resume": True}
