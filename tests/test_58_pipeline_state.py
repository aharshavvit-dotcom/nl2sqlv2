"""
Purpose: Protects feedback unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from orchestration.pipeline_state import PipelineState


def test_state_file_written_loaded_and_step_updated(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = PipelineState(path)
    state.update_step("audit_self_training", "completed", {"ok": True})

    loaded = PipelineState(path)
    payload = loaded.load()

    assert payload["steps"]["audit_self_training"]["status"] == "completed"
    assert payload["last_completed_step"] == "audit_self_training"
