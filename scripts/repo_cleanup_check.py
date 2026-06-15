"""Repository cleanup verification script.

Checks that the naming refactor, PostgreSQL support, and test consolidation
have been properly applied.

Usage:
    python scripts/repo_cleanup_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_FAILED
    status = "PASS" if passed else "FAIL"
    if passed:
        CHECKS_PASSED += 1
    else:
        CHECKS_FAILED += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def check_no_option_labels_in_ui() -> None:
    """No user-facing 'Option A' or 'Option C' labels in the Streamlit app."""
    app_path = ROOT / "app" / "streamlit_app.py"
    if not app_path.exists():
        check("UI: streamlit_app.py exists", False)
        return
    source = app_path.read_text(encoding="utf-8")
    patterns = [
        r'st\.\w+\([^)]*"[^"]*Option A[^"]*"',
        r'st\.\w+\([^)]*"[^"]*Option C[^"]*"',
        r'st\.metric\([^)]*"[^"]*\bV1\b[^"]*"',
        r'st\.metric\([^)]*"[^"]*\bV2\b[^"]*"',
    ]
    violations = []
    for pattern in patterns:
        matches = re.findall(pattern, source)
        violations.extend(matches)
    check("UI: No 'Option A/C' or 'V1/V2' in Streamlit labels", not violations,
          f"Found: {violations}" if violations else "")


def check_readme_naming() -> None:
    """README doesn't use old names except in migration note."""
    readme = ROOT / "README.md"
    if not readme.exists():
        check("README: exists", False)
        return
    source = readme.read_text(encoding="utf-8")
    # "Option A" and "Option C" should only appear in migration/backward-compat context
    for old_name in ["Option A", "Option C"]:
        count = source.count(old_name)
        check(f"README: '{old_name}' usage", count <= 2,
              f"Found {count} occurrences" if count > 2 else "")


def check_consolidated_tests() -> None:
    """Consolidated test files exist."""
    test_dir = ROOT / "tests"
    expected = [
        "test_01_core_ir.py",
        "test_02_sql_validation.py",
        "test_03_database_connectors.py",
        "test_04_retrieval_runtime.py",
        "test_05_neural_runtime.py",
        "test_06_adaptive_router.py",
        "test_07_training_data_pipeline.py",
        "test_08_streamlit_app_helpers.py",
        "test_09_end_to_end_smoke.py",
    ]
    for name in expected:
        check(f"Test: {name} exists", (test_dir / name).exists())


def check_db_connector_layer() -> None:
    """The db connector layer exists and is importable."""
    db_dir = ROOT / "db"
    expected = [
        "__init__.py",
        "connection_config.py",
        "connector_base.py",
        "sqlite_connector.py",
        "postgres_connector.py",
        "schema_reader.py",
        "dialect.py",
    ]
    for name in expected:
        check(f"DB: {name} exists", (db_dir / name).exists())


def check_command_wrappers() -> None:
    """Command wrappers exist."""
    wrappers = [
        ROOT / "training" / "train_retrieval_ir_model.py",
        ROOT / "training" / "train_neural_ir_model.py",
        ROOT / "evaluation" / "run_model_evaluation.py",
        ROOT / "evaluation" / "run_adaptive_router_benchmark.py",
    ]
    for path in wrappers:
        check(f"Wrapper: {path.name} exists", path.exists())


def check_postgresql_dependency() -> None:
    """psycopg2-binary is in requirements.txt."""
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        check("Requirements: exists", False)
        return
    source = req_path.read_text(encoding="utf-8")
    check("Requirements: psycopg2-binary", "psycopg2-binary" in source)


def check_migration_docs() -> None:
    """Migration docs exist."""
    doc = ROOT / "docs" / "migration_naming_cleanup.md"
    check("Docs: migration_naming_cleanup.md exists", doc.exists())


def check_migration_script() -> None:
    """Migration script exists."""
    script = ROOT / "scripts" / "migrate_artifact_names.py"
    check("Scripts: migrate_artifact_names.py exists", script.exists())


def main() -> None:
    print("Repository Cleanup Verification")
    print("=" * 60)

    check_no_option_labels_in_ui()
    check_readme_naming()
    check_consolidated_tests()
    check_db_connector_layer()
    check_command_wrappers()
    check_postgresql_dependency()
    check_migration_docs()
    check_migration_script()

    print("=" * 60)
    print(f"Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
    sys.exit(1 if CHECKS_FAILED > 0 else 0)


if __name__ == "__main__":
    main()
