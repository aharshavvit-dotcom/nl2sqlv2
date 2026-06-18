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
    "self_training/self_improvement_loop.py",
    "execution_eval/sql_canonicalizer.py",
    "execution_eval/sql_structure_comparator.py",
    "execution_eval/result_comparator.py",
    "execution_eval/execution_matcher.py",
    "execution_eval/execution_reporter.py",
    "model_selection/model_candidate.py",
    "model_selection/model_selector.py",
    "model_selection/promotion_policy.py",
    "model_selection/champion_challenger.py",
    "model_selection/selection_reporter.py",
    "orchestration/pipeline_config.py",
    "orchestration/pipeline_runner.py",
    "orchestration/step_runner.py",
    "orchestration/pipeline_state.py",
    "orchestration/pipeline_reporter.py",
    "training/run_execution_aware_evaluation.py",
    "training/select_best_model.py",
    "training/promote_model_if_better.py",
    "training/run_full_training_pipeline.py",
    "pipeline_configs/smoke_training.yaml",
    "pipeline_configs/full_generic_training.yaml",
    "evaluation/model_quality_thresholds.yaml",
]

REQUIRED_COMMANDS = [
    "python scripts/audit_execution_pipeline_readiness.py",
    "python training/run_execution_aware_evaluation.py",
    "python training/select_best_model.py",
    "python training/promote_model_if_better.py",
    "python training/run_full_training_pipeline.py",
    "python training/generate_connected_db_regressions.py",
    "python training/run_connected_db_regressions.py",
]


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.missing_files: list[str] = []
        self.missing_commands: list[str] = []
        self.integration_issues: list[str] = []
        self.recommended_fixes: list[str] = []

    def add(self, check_id: str, name: str, status: str, details: str, required_fix: str = "") -> None:
        self.checks.append({"check_id": check_id, "name": name, "status": status, "details": details, "required_fix": required_fix})
        if status == "fail" and required_fix:
            self.recommended_fixes.append(required_fix)

    def report(self) -> dict[str, Any]:
        summary = {
            "passed": sum(1 for check in self.checks if check["status"] == "pass"),
            "failed": sum(1 for check in self.checks if check["status"] == "fail"),
            "warnings": sum(1 for check in self.checks if check["status"] == "warning"),
        }
        return {
            "overall_status": "fail" if summary["failed"] else ("partial" if summary["warnings"] else "pass"),
            "summary": summary,
            "checks": self.checks,
            "missing_files": sorted(set(self.missing_files)),
            "missing_commands": sorted(set(self.missing_commands)),
            "integration_issues": sorted(set(self.integration_issues)),
            "recommended_fixes": list(dict.fromkeys(self.recommended_fixes)),
        }


def run_audit() -> dict[str, Any]:
    audit = Audit()
    _required_files_check(audit)
    _readme_check(audit)
    _execution_correctness_check(audit)
    _promotion_check(audit)
    _smoke_pipeline_check(audit)
    report = audit.report()
    _write(report)
    return report


def _required_files_check(audit: Audit) -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    audit.missing_files.extend(missing)
    audit.add(
        "EXEC_PIPELINE_001",
        "Required execution pipeline files exist",
        "fail" if missing else "pass",
        "Missing: " + ", ".join(missing) if missing else f"All {len(REQUIRED_FILES)} required files exist.",
        "Create missing execution-evaluation, model-selection, orchestration, CLI, config, or threshold files.",
    )


def _readme_check(audit: Audit) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    missing = [command for command in REQUIRED_COMMANDS if command not in readme]
    audit.missing_commands.extend(missing)
    audit.add(
        "EXEC_PIPELINE_002",
        "README commands are current",
        "warning" if missing else "pass",
        "Missing commands: " + ", ".join(missing) if missing else "README includes current execution and connected-DB commands.",
        "Update README with execution pipeline and connected-DB regression commands.",
    )


def _execution_correctness_check(audit: Audit) -> None:
    try:
        from execution_eval.execution_matcher import ExecutionMatcher

        class Connector:
            def execute(self, sql: str) -> list[dict[str, int]]:
                return [{"id": 1}]

        schema = {
            "tables": {
                "users": {"columns": {"id": {"type": "integer", "primary_key": True}}},
                "assignments": {"columns": {"id": {"type": "integer"}, "user_id": {"type": "integer"}}},
            },
            "relationships": [{"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"}],
        }
        result = ExecutionMatcher().evaluate_example(
            "SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100",
            "SELECT users.id FROM users LIMIT 100",
            schema,
            Connector(),
            "sqlite",
        )
        ok = result.get("execution_match") is True and result.get("correct") is False
    except Exception as exc:
        audit.integration_issues.append(str(exc))
        ok = False
    audit.add(
        "EXEC_PIPELINE_003",
        "Execution success alone is not correctness",
        "pass" if ok else "fail",
        "Execution match requires structural match before correctness." if ok else "Execution-aware matcher may treat result match alone as correct.",
        "Require result match and SQL structure match before marking predictions correct.",
    )


def _promotion_check(audit: Audit) -> None:
    try:
        from model_selection.promotion_policy import PromotionPolicy

        decision = PromotionPolicy().can_promote(
            {"unsafe_sql_count": 1, "sql_validation_rate": 1.0, "no_select_star_rate": 1.0, "unnecessary_join_rate": 0.0, "wrong_table_rate": 0.0},
            {"gold_comparison_score": 1.0, "simple_query_pass_rate": 1.0, "unnecessary_join_rate": 0.0, "unseen_db_sql_validation_rate": 1.0},
            {"unsafe_sql_count_max": 0, "sql_validation_rate": 0.9, "no_select_star_rate": 1.0, "unnecessary_join_rate_max": 0.05, "wrong_table_rate_max": 0.1},
        )
        unsafe_blocked = "unsafe_sql_count" in decision.get("blocking_issues", [])
        regression = PromotionPolicy().can_promote(
            {"unsafe_sql_count": 0, "sql_validation_rate": 1.0, "no_select_star_rate": 1.0, "unnecessary_join_rate": 0.1, "wrong_table_rate": 0.0, "simple_query_pass_rate": 0.8, "gold_comparison_score": 0.8, "unseen_db_sql_validation_rate": 1.0},
            {"gold_comparison_score": 0.9, "simple_query_pass_rate": 1.0, "unnecessary_join_rate": 0.0, "unseen_db_sql_validation_rate": 1.0},
            {"unsafe_sql_count_max": 0, "sql_validation_rate": 0.9, "no_select_star_rate": 1.0, "unnecessary_join_rate_max": 0.2, "wrong_table_rate_max": 0.1, "model_promotion_min_improvement": 0.01},
        )
        regression_blocked = any("regression" in issue for issue in regression.get("blocking_issues", []))
        ok = unsafe_blocked and regression_blocked
    except Exception as exc:
        audit.integration_issues.append(str(exc))
        ok = False
    audit.add(
        "EXEC_PIPELINE_004",
        "Promotion blocks unsafe or regressed models",
        "pass" if ok else "fail",
        "Promotion policy blocks unsafe SQL and quality regressions." if ok else "Promotion policy failed unsafe/regression checks.",
        "Tighten model promotion hard blockers and regression comparisons.",
    )


def _smoke_pipeline_check(audit: Audit) -> None:
    try:
        from orchestration.pipeline_config import PipelineConfig
        from orchestration.step_runner import StepRunner

        config = PipelineConfig.load(ROOT / "pipeline_configs/smoke_training.yaml")
        runner = StepRunner()
        missing_handlers = [step for step in config.steps if getattr(runner, f"_run_{step}", None) is None and step not in {"build_generic_ir_corpus", "build_retrieval_rag_index", "train_neural_ir_model"}]
        ok = config.smoke and not missing_handlers and "run_app_smoke_check" in config.steps
    except Exception as exc:
        audit.integration_issues.append(str(exc))
        missing_handlers = []
        ok = False
    audit.add(
        "EXEC_PIPELINE_005",
        "Smoke pipeline is runnable",
        "pass" if ok else "fail",
        "Smoke config loads and all lightweight steps have runners." if ok else "Missing step handlers: " + ", ".join(missing_handlers),
        "Update smoke pipeline config or StepRunner handlers.",
    )


def _write(report: dict[str, Any]) -> None:
    output = ROOT / "artifacts/audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "execution_pipeline_readiness_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Execution Pipeline Readiness Audit", "", f"Overall status: **{report['overall_status']}**", "", "## Checks"]
    for check in report["checks"]:
        lines.append(f"- **{check['check_id']} {check['name']}**: {check['status']} - {check['details']}")
    (output / "execution_pipeline_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = run_audit()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
