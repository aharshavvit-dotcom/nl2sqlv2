from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_generic_nl2sql_readiness import run_audit


def test_audit_report_created() -> None:
    report = run_audit()
    path = Path("artifacts/audit/generic_nl2sql_readiness_report.json")

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overall_status"] in {"pass", "fail"}
    assert "checks" in report
