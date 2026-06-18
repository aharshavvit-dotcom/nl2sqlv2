from __future__ import annotations

from quality_gates.release_checker import ReleaseChecker


def test_release_readiness_blocks_failed_audit(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "python scripts/audit_generic_nl2sql_readiness.py\n"
        "python training/build_feedback_training_data.py\n"
        "python training/run_release_readiness_check.py\n",
        encoding="utf-8",
    )
    report = ReleaseChecker().evaluate(
        audit_report={"overall_status": "fail", "checks": [{"name": "Direct planner", "status": "fail"}]},
        evaluation_report={},
        quality_gate_report={"passed": True, "metrics": {"no_select_star_rate": 1.0, "unsafe_sql_count_max": 0}},
        regression_report={"passed": True},
        repo_root=tmp_path,
    )

    assert report["release_ready"] is False
    assert any("Audit readiness" in issue for issue in report["blocking_issues"])
