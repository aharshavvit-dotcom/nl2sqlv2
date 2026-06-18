from __future__ import annotations

from pathlib import Path

from orchestration.pipeline_runner import PipelineRunner


def test_steps_run_in_order_state_saved_failure_stops_and_resume(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
pipeline_name: unit_pipeline
smoke: true
skip_heavy_steps: true
steps:
  - run_app_smoke_check
  - unknown_step
""",
        encoding="utf-8",
    )
    state_path = tmp_path / "state.json"
    runner = PipelineRunner(state_path)
    report = runner.run(str(config))

    assert report["status"] == "failed"
    assert [step["step"] for step in report["steps"]] == ["run_app_smoke_check", "unknown_step"]
    assert report["steps"][-1]["status"] == "failed"
    assert state_path.exists()

    resumed = runner.run(str(config), start_at="unknown_step")
    assert resumed["steps"][0]["step"] == "unknown_step"
    assert resumed["status"] == "failed"
