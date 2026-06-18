from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_FILES = [
    "self_training/gold_comparator.py",
    "self_training/error_classifier.py",
    "evaluation/error_taxonomy.yaml",
    "self_training/prediction_runner.py",
    "training/evaluate_against_gold.py",
    "self_training/correction_builder.py",
    "self_training/hard_negative_miner.py",
    "training/build_corrections_from_gold.py",
    "training/mine_validation_errors.py",
    "self_training/dataset_reward_scorer.py",
    "self_training/candidate_generator.py",
    "self_training/ranking_trainer.py",
    "training/train_ranking_from_gold.py",
    "self_training/self_improvement_loop.py",
    "self_training/iteration_reporter.py",
    "training/run_self_improvement_loop.py",
    "execution_eval/sql_canonicalizer.py",
    "execution_eval/sql_structure_comparator.py",
    "execution_eval/result_comparator.py",
    "execution_eval/execution_matcher.py",
    "training/run_execution_aware_evaluation.py",
    "model_selection/model_selector.py",
    "model_selection/promotion_policy.py",
    "orchestration/pipeline_runner.py",
    "training/run_full_training_pipeline.py",
]

REQUIRED_COMMANDS = [
    "python scripts/audit_self_training_readiness.py",
    "python training/evaluate_against_gold.py",
    "python training/mine_validation_errors.py",
    "python training/build_corrections_from_gold.py",
    "python training/train_ranking_from_gold.py",
    "python training/run_self_improvement_loop.py",
    "python training/run_execution_aware_evaluation.py",
    "python training/select_best_model.py",
    "python training/promote_model_if_better.py",
    "python training/run_full_training_pipeline.py",
]


class Audit:
    def __init__(self):
        self.checks: list[dict[str, Any]] = []
        self.missing_files: list[str] = []
        self.missing_commands: list[str] = []
        self.integration_issues: list[str] = []
        self.manual_feedback_issues: list[str] = []
        self.recommended_fixes: list[str] = []

    def add(self, check_id: str, name: str, status: str, details: str, required_fix: str = "") -> None:
        self.checks.append({"check_id": check_id, "name": name, "status": status, "details": details, "required_fix": required_fix})
        if status == "fail" and required_fix:
            self.recommended_fixes.append(required_fix)

    def report(self) -> dict[str, Any]:
        summary = {
            "passed": sum(1 for row in self.checks if row["status"] == "pass"),
            "failed": sum(1 for row in self.checks if row["status"] == "fail"),
            "warnings": sum(1 for row in self.checks if row["status"] == "warning"),
        }
        overall = "fail" if summary["failed"] else ("partial" if summary["warnings"] else "pass")
        return {
            "overall_status": overall,
            "summary": summary,
            "checks": self.checks,
            "missing_files": sorted(set(self.missing_files)),
            "missing_commands": sorted(set(self.missing_commands)),
            "integration_issues": sorted(set(self.integration_issues)),
            "manual_feedback_issues": sorted(set(self.manual_feedback_issues)),
            "recommended_fixes": list(dict.fromkeys(self.recommended_fixes)),
        }


def run_audit() -> dict[str, Any]:
    audit = Audit()
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    audit.missing_files.extend(missing)
    audit.add("SELF_TRAINING_001", "Required self-training files exist", "fail" if missing else "pass", "Missing: " + ", ".join(missing) if missing else f"All {len(REQUIRED_FILES)} required files exist.", "Create missing self-training, execution, model-selection, and orchestration files.")
    _manual_feedback_check(audit)
    _gold_comparator_check(audit)
    _error_taxonomy_check(audit)
    _execution_check(audit)
    _readme_check(audit)
    _write(report := audit.report())
    return report


def _manual_feedback_check(audit: Audit) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    feedback_init = (ROOT / "feedback/__init__.py").read_text(encoding="utf-8") if (ROOT / "feedback/__init__.py").exists() else ""
    app = (ROOT / "app/streamlit_app.py").read_text(encoding="utf-8") if (ROOT / "app/streamlit_app.py").exists() else ""
    issues = []
    if "dataset-driven" not in readme.lower() or "gold" not in readme.lower():
        issues.append("README does not clearly describe dataset-driven gold learning as primary.")
    if "optional" not in feedback_init.lower():
        issues.append("feedback package does not declare manual feedback optional.")
    if "Optional: Manual Feedback" not in app:
        issues.append("Streamlit feedback UI is not marked optional.")
    audit.manual_feedback_issues.extend(issues)
    audit.add("SELF_TRAINING_002", "Manual feedback is optional, not primary", "fail" if issues else "pass", "; ".join(issues) if issues else "Manual feedback is optional and dataset-driven gold learning is primary.", "Make feedback optional and update README/app text.")


def _gold_comparator_check(audit: Audit) -> None:
    try:
        from self_training.gold_comparator import GoldComparator

        result = GoldComparator().compare(
            {"query_ir": {"intent": "show_records", "base_table": "users", "joins": []}, "sql": "SELECT users.id FROM users LIMIT 100"},
            {"query_ir": {"intent": "show_records", "base_table": "users", "joins": []}, "source_sql": "SELECT users.id FROM users LIMIT 100"},
            {"tables": {"users": {"columns": {"id": {}}}}},
        )
        ok = "gold_comparison_score" in result and result.get("execution_success_alone_correct") is False
    except Exception as exc:
        audit.integration_issues.append(str(exc))
        ok = False
    audit.add("SELF_TRAINING_003", "Gold comparator behavior", "pass" if ok else "fail", "GoldComparator.compare checks QueryIR and SQL structure." if ok else "GoldComparator.compare failed minimal behavior.", "Implement GoldComparator.compare with structural and execution-aware fields.")


def _error_taxonomy_check(audit: Audit) -> None:
    text = (ROOT / "evaluation/error_taxonomy.yaml").read_text(encoding="utf-8") if (ROOT / "evaluation/error_taxonomy.yaml").exists() else ""
    required = ["wrong_intent", "unnecessary_join", "invalid_sql", "unsafe_sql", "result_mismatch"]
    missing = [item for item in required if item not in text]
    audit.add("SELF_TRAINING_004", "Error taxonomy exists", "fail" if missing else "pass", "Missing taxonomy entries: " + ", ".join(missing) if missing else "Error taxonomy includes required categories.", "Update evaluation/error_taxonomy.yaml.")


def _execution_check(audit: Audit) -> None:
    try:
        from execution_eval.sql_structure_comparator import SQLStructureComparator

        comparison = SQLStructureComparator().compare(
            "SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100",
            "SELECT users.id FROM users LIMIT 100",
            {"tables": {"users": {"columns": {"id": {}}}, "assignments": {"columns": {"user_id": {}}}}},
        )
        ok = "unnecessary_join" in comparison["errors"]
    except Exception as exc:
        audit.integration_issues.append(str(exc))
        ok = False
    audit.add("SELF_TRAINING_005", "Execution-aware structural comparison", "pass" if ok else "fail", "Unnecessary join detection works." if ok else "Unnecessary join detection failed.", "Fix execution_eval/sql_structure_comparator.py.")


def _readme_check(audit: Audit) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    missing = [command for command in REQUIRED_COMMANDS if command not in readme]
    audit.missing_commands.extend(missing)
    audit.add("SELF_TRAINING_006", "README documents self-training commands", "warning" if missing else "pass", "Missing commands: " + ", ".join(missing) if missing else "README includes required self-training commands.", "Update README final pipeline commands.")


def _write(report: dict[str, Any]) -> None:
    output = ROOT / "artifacts/audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "self_training_readiness_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Self-Training Readiness Audit", "", f"Overall status: **{report['overall_status']}**", "", "## Checks"]
    for check in report["checks"]:
        lines.append(f"- **{check['check_id']} {check['name']}**: {check['status']} - {check['details']}")
    (output / "self_training_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = run_audit()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
