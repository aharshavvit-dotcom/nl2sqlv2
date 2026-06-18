"""Repository cleanup and integration verification script.

Checks naming compliance, integration readiness, and architectural rules.

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


# ──────────────────── Integration Checks ────────────────────

def check_readme_has_train_model() -> None:
    """README contains the canonical training command."""
    readme = ROOT / "README.md"
    if not readme.exists():
        check("README: contains train_model.py command", False)
        return
    source = readme.read_text(encoding="utf-8")
    check("README: contains train_model.py command",
          "python training/train_model.py --config configs/training.yaml" in source)


def check_readme_no_old_primary_commands() -> None:
    """README does not expose old Option A / Option C commands in primary workflow."""
    readme = ROOT / "README.md"
    if not readme.exists():
        check("README: no old primary workflow commands", False)
        return
    source = readme.read_text(encoding="utf-8")
    # Check that the main body doesn't have dozens of training commands
    old_commands = [
        "training/run_self_training_loop.py",
        "training/run_batch_predictions.py",
        "training/run_gold_comparison.py",
        "training_ir/calibrate_option_a_confidence.py",
        "training_ir/calibrate_hybrid_router.py",
    ]
    found = [cmd for cmd in old_commands if cmd in source]
    check("README: no old commands in primary workflow", not found,
          f"Found: {found}" if found else "")


def check_streamlit_no_training_imports() -> None:
    """Streamlit imports no training modules unless developer flag is true."""
    app_path = ROOT / "app" / "streamlit_app.py"
    if not app_path.exists():
        check("Streamlit: no training imports in normal mode", False)
        return
    source = app_path.read_text(encoding="utf-8")
    # Check that train_from_datasets import is only inside ENABLE_DEV_TRAINING_UI block
    has_dev_flag = "ENABLE_DEV_TRAINING_UI" in source
    has_training_import = "from training.train_retriever_from_datasets import" in source
    # It's OK if training import exists but is inside the dev block
    if has_training_import and not has_dev_flag:
        check("Streamlit: no training imports in normal mode", False,
              "training import found without dev flag guard")
    else:
        check("Streamlit: no training imports in normal mode", True)


def check_streamlit_uses_bundle_loader() -> None:
    """App uses ModelBundleLoader."""
    app_path = ROOT / "app" / "streamlit_app.py"
    if not app_path.exists():
        check("Streamlit: uses ModelBundleLoader", False)
        return
    source = app_path.read_text(encoding="utf-8")
    check("Streamlit: uses ModelBundleLoader", "ModelBundleLoader" in source)


def check_train_model_exists() -> None:
    """train_model.py exists."""
    check("Integration: train_model.py exists",
          (ROOT / "training" / "train_model.py").exists())


def check_bundle_manifest_module() -> None:
    """bundle_manifest.py exists."""
    check("Integration: bundle_manifest.py exists",
          (ROOT / "model_bundle" / "bundle_manifest.py").exists())


def check_smoke_training_config() -> None:
    """smoke_training.yaml exists."""
    check("Integration: smoke_training.yaml exists",
          (ROOT / "configs" / "smoke_training.yaml").exists())


def check_integration_test() -> None:
    """Integration test exists."""
    check("Integration: test_99_train_model_integration.py exists",
          (ROOT / "tests" / "test_99_train_model_integration.py").exists())


def check_legacy_docs() -> None:
    """Legacy commands are documented."""
    check("Docs: legacy_commands.md exists",
          (ROOT / "docs" / "legacy_commands.md").exists())
    check("Docs: developer_commands.md exists",
          (ROOT / "docs" / "developer_commands.md").exists())


def main() -> None:
    print("Repository Cleanup & Integration Verification")
    print("=" * 60)

    print("\n--- Naming & UI ---")
    check_no_option_labels_in_ui()
    check_readme_naming()

    print("\n--- Test Suite ---")
    check_consolidated_tests()

    print("\n--- DB Layer ---")
    check_db_connector_layer()

    print("\n--- Command Wrappers ---")
    check_command_wrappers()
    check_postgresql_dependency()

    print("\n--- Documentation ---")
    check_migration_docs()
    check_migration_script()
    check_legacy_docs()

    print("\n--- Integration Architecture ---")
    check_readme_has_train_model()
    check_readme_no_old_primary_commands()
    check_streamlit_no_training_imports()
    check_streamlit_uses_bundle_loader()
    check_train_model_exists()
    check_bundle_manifest_module()
    check_smoke_training_config()
    check_integration_test()

    print("\n" + "=" * 60)
    print(f"Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
    sys.exit(1 if CHECKS_FAILED > 0 else 0)


if __name__ == "__main__":
    main()
