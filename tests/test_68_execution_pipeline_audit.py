from __future__ import annotations

from scripts.audit_execution_pipeline_readiness import run_audit


def test_execution_pipeline_audit_writes_expected_report() -> None:
    report = run_audit()

    assert report["overall_status"] in {"pass", "partial"}
    assert not report["missing_files"]
    assert any(check["check_id"] == "EXEC_PIPELINE_003" and check["status"] == "pass" for check in report["checks"])
