"""Audit integration readiness of the NL-to-SQL system.

Checks 15 architectural integration requirements and produces
JSON and Markdown reports.

Usage:
    python scripts/audit_integration_readiness.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_audit() -> dict[str, Any]:
    """Run all integration readiness checks."""
    checks: list[dict[str, Any]] = []
    blocking_issues: list[str] = []
    recommended_fixes: list[str] = []

    # 1. Canonical training command exists
    train_model = ROOT / "training" / "train_model.py"
    checks.append(_check(
        "canonical_training_command",
        "Canonical training command exists",
        train_model.exists(),
        "training/train_model.py exists" if train_model.exists() else "training/train_model.py missing",
    ))
    if not train_model.exists():
        blocking_issues.append("training/train_model.py is missing")
        recommended_fixes.append("Create training/train_model.py as the canonical training entry point")

    # 2. Canonical training config exists
    training_yaml = ROOT / "configs" / "training.yaml"
    checks.append(_check(
        "canonical_training_config",
        "Canonical training config exists",
        training_yaml.exists(),
        "configs/training.yaml exists" if training_yaml.exists() else "configs/training.yaml missing",
    ))
    if not training_yaml.exists():
        blocking_issues.append("configs/training.yaml is missing")

    # 3. Model bundle package exists
    bundle_pkg = ROOT / "model_bundle" / "__init__.py"
    checks.append(_check(
        "model_bundle_package",
        "Model bundle package exists",
        bundle_pkg.exists(),
        "model_bundle/__init__.py exists" if bundle_pkg.exists() else "model_bundle/ package missing",
    ))

    # 4. Bundle manifest schema exists
    manifest_mod = ROOT / "model_bundle" / "bundle_manifest.py"
    checks.append(_check(
        "bundle_manifest_schema",
        "Bundle manifest schema exists",
        manifest_mod.exists(),
        "model_bundle/bundle_manifest.py exists" if manifest_mod.exists() else "bundle_manifest.py missing",
    ))

    # 5. Pipeline contracts exist
    contracts = ROOT / "orchestration" / "step_contract.py"
    checks.append(_check(
        "pipeline_contracts",
        "Pipeline contracts exist",
        contracts.exists(),
        "orchestration/step_contract.py exists" if contracts.exists() else "step_contract.py missing",
    ))

    # 6. Pipeline validates required inputs and outputs
    validator = ROOT / "orchestration" / "contract_validator.py"
    checks.append(_check(
        "pipeline_contract_validator",
        "Pipeline contract validator exists",
        validator.exists(),
        "orchestration/contract_validator.py exists" if validator.exists() else "contract_validator.py missing",
    ))

    # 7. Streamlit loads model bundle instead of guessing artifacts
    app_path = ROOT / "app" / "streamlit_app.py"
    streamlit_uses_bundle = False
    if app_path.exists():
        source = app_path.read_text(encoding="utf-8")
        streamlit_uses_bundle = "ModelBundleLoader" in source or "bundle_loader" in source
    checks.append(_check(
        "streamlit_loads_bundle",
        "Streamlit loads model bundle",
        streamlit_uses_bundle,
        "ModelBundleLoader found in streamlit_app.py" if streamlit_uses_bundle else "Streamlit does not use ModelBundleLoader",
    ))

    # 8. Streamlit normal UI does not train models
    streamlit_no_training = False
    if app_path.exists():
        source = app_path.read_text(encoding="utf-8")
        has_dev_flag = "ENABLE_DEV_TRAINING_UI" in source
        # Check that the "Train From Local Datasets" button is gated
        has_train_button = 'st.button("Train From Local Datasets")' in source
        if has_dev_flag:
            streamlit_no_training = True
        elif not has_train_button:
            streamlit_no_training = True
    checks.append(_check(
        "streamlit_no_training_ui",
        "Streamlit normal UI does not train models",
        streamlit_no_training,
        "Training UI gated by ENABLE_DEV_TRAINING_UI" if streamlit_no_training else "Training UI exposed in normal mode",
    ))

    # 9. README has one main training command
    readme_path = ROOT / "README.md"
    readme_has_main = False
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        readme_has_main = "python training/train_model.py --config configs/training.yaml" in readme
    checks.append(_check(
        "readme_main_training_command",
        "README has one main training command",
        readme_has_main,
        "train_model.py command found in README" if readme_has_main else "README missing canonical training command",
    ))

    # 10. Legacy commands are not shown as primary workflow
    legacy_not_primary = True
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        old_cmds = [
            "training/run_self_training_loop.py",
            "training/run_batch_predictions.py",
            "training_ir/calibrate_option_a_confidence.py",
        ]
        found = [cmd for cmd in old_cmds if cmd in readme]
        legacy_not_primary = not found
    checks.append(_check(
        "legacy_commands_not_primary",
        "Legacy commands not in primary README workflow",
        legacy_not_primary,
        "" if legacy_not_primary else f"Found legacy commands in README: {found}",
    ))

    # 11. Old naming does not appear in active README workflow
    no_old_naming = True
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        old_names_count = readme.count("Option A") + readme.count("Option C")
        no_old_naming = old_names_count <= 2  # Allow in migration note
    checks.append(_check(
        "no_old_naming_in_readme",
        "Old naming not in active README workflow",
        no_old_naming,
        "" if no_old_naming else f"Found {old_names_count} old name references",
    ))

    # 12. train_neural_ir_model.py args are supported or deprecated
    neural_train = ROOT / "training" / "train_neural_ir_model.py"
    neural_args_ok = False
    if neural_train.exists():
        source = neural_train.read_text(encoding="utf-8")
        required_args = ["--train", "--validation", "--hard-negatives", "--output-dir", "--epochs", "--batch-size"]
        neural_args_ok = all(arg in source for arg in required_args)
    checks.append(_check(
        "neural_ir_model_args",
        "train_neural_ir_model.py README arguments supported",
        neural_args_ok,
        "All documented args found" if neural_args_ok else "Some documented args missing from train_neural_ir_model.py",
    ))

    # 13. Smoke integration test exists
    smoke_test = ROOT / "tests" / "test_99_train_model_integration.py"
    checks.append(_check(
        "smoke_integration_test",
        "Smoke integration test exists",
        smoke_test.exists(),
        "test_99_train_model_integration.py exists" if smoke_test.exists() else "Integration test missing",
    ))

    # 14. Quality gate exists
    quality_gate = ROOT / "quality_gates" / "integrated_quality_gate.py"
    checks.append(_check(
        "quality_gate_exists",
        "Integrated quality gate exists",
        quality_gate.exists(),
        "integrated_quality_gate.py exists" if quality_gate.exists() else "Integrated quality gate missing",
    ))

    # 15. Model bundle validation exists
    bundle_val = ROOT / "model_bundle" / "bundle_validator.py"
    checks.append(_check(
        "model_bundle_validation",
        "Model bundle validation exists",
        bundle_val.exists(),
        "bundle_validator.py exists" if bundle_val.exists() else "Bundle validator missing",
    ))

    # 16. Pipeline runner rejects unknown steps
    step_runner = ROOT / "orchestration" / "step_runner.py"
    unknown_steps_fail = False
    if step_runner.exists():
        source = step_runner.read_text(encoding="utf-8")
        unknown_steps_fail = "Unknown pipeline step" in source and "no runner implemented" not in source
    checks.append(_check(
        "pipeline_unknown_steps_fail",
        "Pipeline unknown steps fail",
        unknown_steps_fail,
        "Unknown steps raise errors" if unknown_steps_fail else "Unknown steps may still skip silently",
    ))

    # 17. Neural wrapper defaults to generic corpus
    neural_train = ROOT / "training" / "train_neural_ir_model.py"
    neural_generic_default = False
    if neural_train.exists():
        source = neural_train.read_text(encoding="utf-8")
        neural_generic_default = "generic_ir_train.jsonl" in source and "--legacy" in source and "ir_training_examples.jsonl" not in source
    checks.append(_check(
        "neural_wrapper_generic_default",
        "Neural wrapper uses generic corpus by default",
        neural_generic_default,
        "Optimized generic-corpus path is default" if neural_generic_default else "Old IR training default still visible",
    ))

    # 18. Runtime sample fallback is dev-only
    runtime_model = ROOT / "retriever" / "retrieval_nl2sql_model.py"
    no_sample_fallback = False
    if runtime_model.exists():
        source = runtime_model.read_text(encoding="utf-8")
        no_sample_fallback = "allow_dev_fallback" in source and "No validated model bundle found" in source
    checks.append(_check(
        "runtime_no_sample_fallback",
        "Runtime sample fallback is dev-only",
        no_sample_fallback,
        "Missing artifacts raise bundle error by default" if no_sample_fallback else "Runtime may still silently fall back",
    ))

    # 19. Bundle validator has blocking artifact and dataset checks
    bundle_strict = False
    if bundle_val.exists():
        source = bundle_val.read_text(encoding="utf-8")
        bundle_strict = "Required {label} artifact missing" in source and "Dataset leakage check failed" in source
    checks.append(_check(
        "bundle_validator_strict",
        "Bundle validator has strict checks",
        bundle_strict,
        "Strict artifact and dataset checks found" if bundle_strict else "Bundle validator still appears advisory",
    ))

    # Compute summary
    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    warnings = sum(1 for c in checks if c.get("severity") == "warning")

    if failed == 0:
        overall = "pass"
    elif failed <= 3:
        overall = "partial"
    else:
        overall = "fail"

    report = {
        "overall_status": overall,
        "summary": {
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
        },
        "checks": checks,
        "blocking_issues": blocking_issues,
        "recommended_fixes": recommended_fixes,
    }

    return report


def _check(check_id: str, description: str, passed: bool, detail: str = "", severity: str = "error") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "description": description,
        "passed": passed,
        "detail": detail,
        "severity": severity if not passed else "ok",
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown reports."""
    output_dir = ROOT / "artifacts" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    (output_dir / "integration_readiness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# Integration Readiness Report",
        "",
        f"**Overall Status:** {report['overall_status'].upper()}",
        f"**Passed:** {report['summary']['passed']} | **Failed:** {report['summary']['failed']} | **Warnings:** {report['summary']['warnings']}",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        icon = "✅" if check["passed"] else "❌"
        lines.append(f"- {icon} **{check['description']}**")
        if check.get("detail"):
            lines.append(f"  - {check['detail']}")

    if report["blocking_issues"]:
        lines.extend(["", "## Blocking Issues", ""])
        for issue in report["blocking_issues"]:
            lines.append(f"- ❌ {issue}")

    if report["recommended_fixes"]:
        lines.extend(["", "## Recommended Fixes", ""])
        for fix in report["recommended_fixes"]:
            lines.append(f"- 🔧 {fix}")

    lines.append("")
    (output_dir / "integration_readiness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    print("Integration Readiness Audit")
    print("=" * 60)

    report = run_audit()
    write_reports(report)

    for check in report["checks"]:
        icon = "[PASS]" if check["passed"] else "[FAIL]"
        print(f"  {icon} {check['description']}")

    print()
    print(f"Overall: {report['overall_status'].upper()}")
    print(f"Passed: {report['summary']['passed']}, Failed: {report['summary']['failed']}")
    print(f"\nReports written to:")
    print(f"  artifacts/audit/integration_readiness_report.json")
    print(f"  artifacts/audit/integration_readiness_report.md")

    return 0 if report["overall_status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
