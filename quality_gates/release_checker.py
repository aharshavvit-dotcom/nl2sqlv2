from __future__ import annotations

from pathlib import Path
from typing import Any


class ReleaseChecker:
    def evaluate(
        self,
        audit_report: dict[str, Any],
        evaluation_report: dict[str, Any],
        quality_gate_report: dict[str, Any],
        regression_report: dict[str, Any],
        repo_root: str | Path = ".",
    ) -> dict[str, Any]:
        blocking: list[str] = []
        warnings: list[str] = []
        next_actions: list[str] = []
        root = Path(repo_root)

        if audit_report.get("overall_status") != "pass":
            blocking.append("Audit readiness did not pass.")
            next_actions.append("Run scripts/audit_generic_nl2sql_readiness.py and address failed checks.")
        if quality_gate_report and not quality_gate_report.get("passed", False):
            blocking.append("Model quality gate failed.")
            next_actions.append("Improve model metrics or adjust thresholds with justification.")
        if regression_report and not regression_report.get("passed", False):
            blocking.append("Regression suite failed.")
            next_actions.append("Fix blocking regression cases before release.")

        metrics = quality_gate_report.get("metrics", {}) if isinstance(quality_gate_report, dict) else {}
        if metrics.get("unsafe_sql_count_max", 0) > 0 or evaluation_report.get("unsafe_sql_count", 0) > 0:
            blocking.append("Unsafe SQL was observed.")
        if metrics.get("no_select_star_rate", 1.0) < 1.0:
            blocking.append("SELECT * was observed.")

        failed_audit_checks = [
            check for check in audit_report.get("checks", []) if check.get("status") == "fail"
        ]
        generic_failures = [check for check in failed_audit_checks if "planner" in check.get("name", "").lower()]
        if generic_failures:
            blocking.append("Known generic planner failures remain.")

        if not list(root.glob("artifacts/**/model_manifest.json")):
            warnings.append("No model_manifest.json found under artifacts.")
            next_actions.append("Register trained artifacts with model_registry.ArtifactRegistry.")

        readme = root / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            for command in [
                "scripts/audit_generic_nl2sql_readiness.py",
                "training/build_feedback_training_data.py",
                "training/run_release_readiness_check.py",
            ]:
                if command not in text:
                    warnings.append(f"README is missing command: {command}")
        else:
            blocking.append("README.md is missing.")

        return {
            "release_ready": not blocking,
            "blocking_issues": blocking,
            "warnings": list(dict.fromkeys(warnings)),
            "recommended_next_actions": list(dict.fromkeys(next_actions)),
        }
