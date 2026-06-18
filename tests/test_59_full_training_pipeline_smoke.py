from __future__ import annotations

from pathlib import Path

from orchestration.pipeline_runner import PipelineRunner


def test_smoke_pipeline_runs_with_tiny_mock_data(tmp_path: Path) -> None:
    config = tmp_path / "smoke.yaml"
    config.write_text(
        """
pipeline_name: tiny_smoke
smoke: true
skip_heavy_steps: true
steps:
  - run_app_smoke_check
""",
        encoding="utf-8",
    )
    report = PipelineRunner(tmp_path / "state.json").run(str(config))

    assert report["status"] == "completed"
    assert Path("artifacts/pipeline/pipeline_report.json").exists()
