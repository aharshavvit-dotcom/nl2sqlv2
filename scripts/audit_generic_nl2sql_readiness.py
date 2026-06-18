from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl
from generic_planner import JoinPolicy, SchemaProfile, TableIntentResolver, infer_join_policy
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from ir.ir_to_sql_renderer import IRToSQLRenderer
from retrieval import ExampleIndex, LocalRAGRetriever, PatternIndex, RetrievalReranker, SchemaIndex
from tests.test_10_generic_table_intent import GENERIC_POSTGRES_SCHEMA
from validation.sql_validator import SQLValidator


REQUIRED_GENERIC_PLANNER = [
    "generic_planner/__init__.py",
    "generic_planner/table_intent_resolver.py",
    "generic_planner/schema_text_normalizer.py",
    "generic_planner/direct_queryir_builder.py",
    "generic_planner/join_policy.py",
    "generic_planner/schema_profile.py",
    "generic_planner/generic_slot_resolver.py",
    "generic_planner/planner_result.py",
]

REQUIRED_DATASET_TRAINING = [
    "dataset_training/dataset_registry.py",
    "dataset_training/split_manager.py",
    "dataset_training/schema_splitter.py",
    "dataset_training/ir_corpus_builder.py",
    "dataset_training/corpus_quality.py",
    "dataset_training/retrieval_corpus_builder.py",
    "dataset_training/neural_corpus_builder.py",
    "dataset_training/hard_negative_corpus_builder.py",
    "dataset_training/dataset_evaluator.py",
    "dataset_training/benchmark_runner.py",
    "dataset_training/leakage_checker.py",
    "dataset_training/curriculum_builder.py",
    "dataset_training/reporting.py",
]

REQUIRED_WRAPPERS = [
    "training/build_generic_ir_corpus.py",
    "training/build_retrieval_rag_index.py",
    "training/build_hard_negative_corpus.py",
    "training/train_neural_ir_model.py",
    "training/evaluate_generic_models.py",
    "training/run_unseen_db_benchmark.py",
    "training/build_feedback_training_data.py",
    "training/rebuild_feedback_index.py",
    "training/run_model_quality_gate.py",
    "training/run_regression_suite.py",
    "training/run_release_readiness_check.py",
]

REQUIRED_RETRIEVAL = [
    "retrieval/rag_index_builder.py",
    "retrieval/rag_retriever.py",
    "retrieval/schema_index.py",
    "retrieval/example_index.py",
    "retrieval/pattern_index.py",
    "retrieval/feedback_index.py",
    "retrieval/retrieval_reranker.py",
]

REQUIRED_TESTS = [
    "tests/test_01_core_ir.py",
    "tests/test_02_sql_validation.py",
    "tests/test_03_database_connectors.py",
    "tests/test_04_retrieval_runtime.py",
    "tests/test_05_neural_runtime.py",
    "tests/test_06_adaptive_router.py",
    "tests/test_07_training_data_pipeline.py",
    "tests/test_08_streamlit_app_helpers.py",
    "tests/test_09_end_to_end_smoke.py",
    "tests/test_10_generic_table_intent.py",
    "tests/test_11_generic_join_policy.py",
    "tests/test_12_generic_postgres_schema_runtime.py",
    "tests/test_20_dataset_split_manager.py",
    "tests/test_21_dataset_leakage_checker.py",
    "tests/test_22_generic_ir_corpus_builder.py",
    "tests/test_23_retrieval_rag_index.py",
    "tests/test_24_dataset_scale_evaluator.py",
    "tests/test_25_unseen_db_benchmark.py",
    "tests/test_26_training_command_wrappers.py",
    "tests/test_30_audit_readiness.py",
    "tests/test_31_feedback_store.py",
    "tests/test_32_feedback_to_training_data.py",
    "tests/test_33_feedback_index.py",
    "tests/test_34_reward_scorer.py",
    "tests/test_35_model_quality_gate.py",
    "tests/test_36_regression_suite.py",
    "tests/test_37_model_artifact_registry.py",
    "tests/test_38_release_readiness.py",
]

REQUIRED_ARTIFACTS = [
    "data/processed/generic_ir_train.jsonl",
    "data/processed/generic_ir_validation.jsonl",
    "data/processed/generic_ir_test.jsonl",
    "data/processed/generic_ir_unseen_db_test.jsonl",
    "data/processed/generic_ir_unsupported.jsonl",
    "artifacts/generic_training/dataset_split_report.json",
    "artifacts/generic_training/leakage_report.json",
    "artifacts/generic_training/corpus_quality_report.json",
]


class Audit:
    def __init__(self):
        self.checks: list[dict[str, Any]] = []
        self.missing_files: list[str] = []
        self.stale_files: list[str] = []
        self.naming_issues: list[str] = []
        self.integration_issues: list[str] = []
        self.recommended_fixes: list[str] = []

    def add(self, check_id: str, name: str, status: str, details: str, required_fix: str = "") -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "name": name,
                "status": status,
                "details": details,
                "required_fix": required_fix,
            }
        )
        if status == "fail" and required_fix:
            self.recommended_fixes.append(required_fix)

    def file_check(self, check_id: str, name: str, files: list[str], status_if_missing: str = "fail") -> None:
        missing = [path for path in files if not (ROOT / path).exists()]
        if missing:
            self.missing_files.extend(missing)
            self.add(check_id, name, status_if_missing, "Missing: " + ", ".join(missing), "Create or restore the missing files.")
        else:
            self.add(check_id, name, "pass", f"All {len(files)} required files exist.")

    def report(self) -> dict[str, Any]:
        summary = {
            "passed": sum(1 for check in self.checks if check["status"] == "pass"),
            "failed": sum(1 for check in self.checks if check["status"] == "fail"),
            "warnings": sum(1 for check in self.checks if check["status"] == "warning"),
        }
        return {
            "overall_status": "fail" if summary["failed"] else "pass",
            "summary": summary,
            "checks": self.checks,
            "missing_files": sorted(set(self.missing_files)),
            "stale_files": sorted(set(self.stale_files)),
            "naming_issues": sorted(set(self.naming_issues)),
            "integration_issues": sorted(set(self.integration_issues)),
            "recommended_fixes": list(dict.fromkeys(self.recommended_fixes)),
        }


def audit_direct_planner(audit: Audit) -> None:
    renderer = IRToSQLRenderer()
    validator = SQLValidator()
    resolver = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA))
    cases = [
        ("list all users", "show_records", "users", None),
        ("list all berth_masters", "show_records", "berth_masters", None),
        ("count users", "count_records", "users", "COUNT(*)"),
        ("show users where role is admin", "simple_filter", "users", "role"),
    ]
    failures = []
    for question, expected_intent, expected_table, expected_sql_part in cases:
        result = resolver.resolve(question)
        query_ir = result.query_ir
        if not result.handled or query_ir is None:
            failures.append(f"{question}: not handled ({result.reason})")
            continue
        sql = renderer.render(query_ir, dialect="postgres")
        validation = validator.validate(sql, schema=GENERIC_POSTGRES_SCHEMA, dialect="postgres")
        if query_ir.metadata.get("source") != "generic_direct_planner":
            failures.append(f"{question}: wrong source")
        if query_ir.intent != expected_intent:
            failures.append(f"{question}: expected {expected_intent}, got {query_ir.intent}")
        if query_ir.base_table != expected_table:
            failures.append(f"{question}: expected table {expected_table}, got {query_ir.base_table}")
        if query_ir.joins:
            failures.append(f"{question}: unexpected joins")
        if not validation.get("is_valid"):
            failures.append(f"{question}: invalid SQL {validation.get('issues')}")
        if f'FROM "{expected_table}"' not in sql and f"FROM {expected_table}" not in sql:
            failures.append(f"{question}: SQL missing FROM {expected_table}")
        if "LIMIT" not in sql.upper():
            failures.append(f"{question}: SQL missing LIMIT")
        if "SELECT *" in sql.upper():
            failures.append(f"{question}: SQL used SELECT *")
        if "password_hash" in sql:
            failures.append(f"{question}: SQL selected sensitive column")
        if expected_sql_part and expected_sql_part not in sql:
            failures.append(f"{question}: SQL missing {expected_sql_part}")
    audit.add(
        "GENERIC_PLANNER_001",
        "Direct table planner behavior",
        "fail" if failures else "pass",
        "; ".join(failures) if failures else "Direct show/count/filter cases render safe no-join SQL.",
        "Fix generic_planner direct planning, SQL rendering, or SQL validation.",
    )


def audit_join_policy(audit: Audit) -> None:
    context = RuntimeSchemaContext(GENERIC_POSTGRES_SCHEMA)
    failures = []
    expected = [
        ("list all users", "show_records", JoinPolicy.NONE),
        ("count users", "count_records", JoinPolicy.NONE),
        ("show users where role is admin", "simple_filter", JoinPolicy.NONE),
        ("show assignments with user names", "show_records", JoinPolicy.EXPLICIT_ONLY),
        ("assignments by user", "metric_by_dimension", JoinPolicy.EXPLICIT_ONLY),
    ]
    for question, intent, policy in expected:
        actual = infer_join_policy(question, intent)
        if actual != policy:
            failures.append(f"{question}: expected {policy.value}, got {actual.value}")
    plan = RuntimeJoinPlanner().plan_joins(context, "users", ["users", "assignments"], join_policy=JoinPolicy.NONE)
    if plan.join_steps or plan.join_clause or plan.required_tables != ["users"]:
        failures.append("JoinPolicy.NONE still planned joins.")
    audit.add(
        "JOIN_POLICY_001",
        "Join policy inference and enforcement",
        "fail" if failures else "pass",
        "; ".join(failures) if failures else "NONE and EXPLICIT_ONLY policies are inferred and enforced.",
        "Fix generic_planner/join_policy.py or inference/runtime_join_planner.py.",
    )


def audit_sample_retail_bias(audit: Audit) -> None:
    profile = SchemaProfile(GENERIC_POSTGRES_SCHEMA)
    sales_matches = profile.find_table_matches("sales")
    customer_matches = profile.find_table_matches("customer")
    product_matches = profile.find_table_matches("product")
    user_matches = profile.find_table_matches("list all users")
    failures = []
    if any(match["score"] >= 0.80 for match in [*sales_matches, *customer_matches, *product_matches]):
        failures.append("Retail terms matched strongly against non-retail schema.")
    if not user_matches or user_matches[0]["table"] != "users":
        failures.append("Live schema aliases did not resolve users.")
    audit.add(
        "SCHEMA_BIAS_001",
        "No sample-retail bias for generic schemas",
        "fail" if failures else "pass",
        "; ".join(failures) if failures else "Retail aliases are not globally applied to generic schemas.",
        "Guard retail mappings and generate aliases from the live schema.",
    )


def audit_split_leakage(audit: Audit) -> None:
    train = read_jsonl(ROOT / "data/processed/generic_ir_train.jsonl")
    validation = read_jsonl(ROOT / "data/processed/generic_ir_validation.jsonl")
    unseen = read_jsonl(ROOT / "data/processed/generic_ir_unseen_db_test.jsonl")
    train_dbs = {row.get("db_id") for row in train if row.get("db_id")}
    validation_dbs = {row.get("db_id") for row in validation if row.get("db_id")}
    unseen_dbs = {row.get("db_id") for row in unseen if row.get("db_id")}
    overlap = sorted((train_dbs | validation_dbs) & unseen_dbs)
    if overlap:
        audit.integration_issues.append("Unseen DB split overlaps train/validation DBs.")
    audit.add(
        "SPLIT_001",
        "Database-level unseen split has no leakage",
        "fail" if overlap else "pass",
        f"Overlap db_ids: {overlap[:20]}" if overlap else f"Checked {len(train_dbs)} train, {len(validation_dbs)} validation, {len(unseen_dbs)} unseen DB ids.",
        "Fix dataset_training/split_manager.py and rebuild generic corpus.",
    )


def audit_rag_retrieval(audit: Audit) -> None:
    examples = [
        {
            "example_id": "show_users",
            "question": "list all users",
            "intent": "show_records",
            "template_id": "show_records",
            "query_ir": {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": []},
        },
        {
            "example_id": "top_sales",
            "question": "top customers by sales",
            "intent": "top_n_metric_by_dimension",
            "template_id": "top_n_metric_by_dimension",
            "query_ir": {"intent": "top_n_metric_by_dimension", "template_id": "top_n_metric_by_dimension", "base_table": "orders", "required_tables": ["orders", "customers"], "joins": [{"condition": "orders.customer_id = customers.customer_id"}], "metrics": [{"expression": "orders.amount"}]},
        },
    ]
    example_index = ExampleIndex()
    schema_index = SchemaIndex()
    pattern_index = PatternIndex()
    example_index.build(examples)
    schema_index.build(examples)
    pattern_index.build(examples)
    result = LocalRAGRetriever(example_index, schema_index, pattern_index, RetrievalReranker()).retrieve(
        "list all users",
        {"tables": {"users": {"columns": {"id": {}, "name": {}}}, "assignments": {"columns": {"id": {}}}}},
        top_k=2,
    )
    failures = []
    if not result["patterns"] or result["patterns"][0]["pattern"] != "show_records":
        failures.append("show_records was not the top inferred pattern.")
    if not result["reranked"] or result["reranked"][0].get("example_id") != "show_users":
        failures.append("RAG reranker did not prioritize the show_users example.")
    audit.add(
        "RAG_001",
        "Local RAG retrieval prioritizes direct patterns",
        "fail" if failures else "pass",
        "; ".join(failures) if failures else "Simple listing retrieval avoids metric/join examples.",
        "Fix retrieval/rag_retriever.py or retrieval/retrieval_reranker.py.",
    )


def audit_naming(audit: Audit) -> None:
    targets = [ROOT / "README.md", ROOT / "app/streamlit_app.py"]
    bad_lines = []
    for path in targets:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "migration_naming_cleanup" in line or "old folder names" in line:
                continue
            if re.search(r"\bOption [AC]\b", line):
                bad_lines.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    audit.naming_issues.extend(bad_lines)
    audit.add(
        "NAMING_001",
        "No old user-facing model names",
        "warning" if bad_lines else "pass",
        "; ".join(bad_lines[:10]) if bad_lines else "README and Streamlit labels use production model names.",
        "Replace old user-facing names with Retrieval QueryIR Model, Neural QueryIR Model, and Adaptive QueryIR Router.",
    )


def audit_readme(audit: Audit) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    commands = [
        "python scripts/audit_generic_nl2sql_readiness.py",
        "python training/build_generic_ir_corpus.py",
        "python training/build_retrieval_rag_index.py",
        "python training/train_neural_ir_model.py",
        "python training/evaluate_generic_models.py",
        "python training/build_feedback_training_data.py",
        "python training/rebuild_feedback_index.py",
        "python training/run_model_quality_gate.py",
        "python training/run_regression_suite.py",
        "python training/run_release_readiness_check.py",
        "streamlit run app/streamlit_app.py",
    ]
    missing = [command for command in commands if command not in readme]
    audit.add(
        "README_001",
        "README has stepwise production commands",
        "fail" if missing else "pass",
        "Missing commands: " + ", ".join(missing) if missing else "README includes all required stepwise commands.",
        "Update README with audit, training, feedback, quality-gate, regression, release, and Streamlit commands.",
    )


def write_reports(report: dict[str, Any]) -> None:
    output = ROOT / "artifacts" / "audit"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "generic_nl2sql_readiness_report.json"
    md_path = output / "generic_nl2sql_readiness_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Generic NL-to-SQL Readiness Audit", "", f"Overall status: **{report['overall_status']}**", ""]
    lines.append("## Summary")
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Checks"])
    for check in report["checks"]:
        lines.append(f"- **{check['check_id']} {check['name']}**: {check['status']} - {check['details']}")
    if report["recommended_fixes"]:
        lines.extend(["", "## Recommended Fixes"])
        lines.extend(f"- {item}" for item in report["recommended_fixes"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit() -> dict[str, Any]:
    audit = Audit()
    audit.file_check("FILES_001", "Generic planner files exist", REQUIRED_GENERIC_PLANNER)
    audit.file_check("FILES_002", "Dataset-scale training files exist", REQUIRED_DATASET_TRAINING)
    audit.file_check("FILES_003", "Retrieval RAG files exist", REQUIRED_RETRIEVAL)
    audit.file_check("FILES_004", "Training and quality CLI wrappers exist", REQUIRED_WRAPPERS)
    audit.file_check("FILES_005", "Consolidated tests exist", REQUIRED_TESTS)
    audit.file_check("ARTIFACTS_001", "Required generated training artifacts exist", REQUIRED_ARTIFACTS, status_if_missing="warning")
    audit_direct_planner(audit)
    audit_join_policy(audit)
    audit_sample_retail_bias(audit)
    audit_split_leakage(audit)
    audit_rag_retrieval(audit)
    audit_naming(audit)
    audit_readme(audit)
    report = audit.report()
    write_reports(report)
    return report


def main() -> int:
    report = run_audit()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
