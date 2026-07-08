"""Report identity validation for bundle governance.

Ensures all reports attached to a bundle share the same pipeline_run_id
and that post-bundle reports match the bundle_id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def validate_report_identity(
    report: dict[str, Any],
    *,
    expected_pipeline_run_id: str | None = None,
    expected_bundle_id: str | None = None,
    is_post_bundle: bool = False,
    report_name: str = "unknown",
) -> dict[str, Any]:
    """Validate a single report's identity fields.

    Args:
        report: The report dict to validate.
        expected_pipeline_run_id: If set, the report must have this run ID.
        expected_bundle_id: If set and is_post_bundle, the report must match.
        is_post_bundle: If True, applies stricter bundle_id matching.
        report_name: Human-readable name for error messages.

    Returns:
        Dict with 'valid', 'issues', and extracted identity fields.
    """
    issues: list[str] = []
    report_run_id = report.get("pipeline_run_id")
    report_bundle_id = report.get("bundle_id")
    report_generated_at = report.get("generated_at")

    # Check pipeline_run_id
    if expected_pipeline_run_id:
        if not report_run_id:
            issues.append(f"{report_name}: missing pipeline_run_id")
        elif str(report_run_id) != str(expected_pipeline_run_id):
            issues.append(
                f"{report_name}: pipeline_run_id mismatch: "
                f"report={report_run_id} expected={expected_pipeline_run_id}"
            )

    # Post-bundle reports must match bundle_id
    if is_post_bundle and expected_bundle_id:
        if not report_bundle_id:
            issues.append(f"{report_name}: post-bundle report missing bundle_id")
        elif str(report_bundle_id) != str(expected_bundle_id):
            issues.append(
                f"{report_name}: bundle_id mismatch: "
                f"report={report_bundle_id} expected={expected_bundle_id}"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "report_name": report_name,
        "pipeline_run_id": report_run_id,
        "bundle_id": report_bundle_id,
        "generated_at": report_generated_at,
    }


def validate_bundle_report_identities(
    bundle_dir: str | Path,
    *,
    expected_pipeline_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate identity consistency across all reports in a bundle.

    Checks that all reports share the same pipeline_run_id and that
    post-bundle reports match the bundle_id from the manifest.

    Args:
        bundle_dir: Path to the candidate or current bundle directory.
        expected_pipeline_run_id: If set, all reports must have this run ID.

    Returns:
        Dict with 'valid', 'issues', 'warnings', and per-report results.
    """
    bundle_path = Path(bundle_dir)
    issues: list[str] = []
    warnings: list[str] = []
    report_results: list[dict[str, Any]] = []

    # Load manifest
    manifest_path = bundle_path / "bundle_manifest.json"
    if not manifest_path.exists():
        return {
            "valid": False,
            "issues": ["bundle_manifest.json not found"],
            "warnings": [],
            "reports": [],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_id = manifest.get("bundle_id", "")
    manifest_run_id = manifest.get("pipeline_run_id", "")
    manifest_created_at = manifest.get("created_at", "")

    # Use manifest run ID if no explicit expected ID provided
    if not expected_pipeline_run_id and manifest_run_id:
        expected_pipeline_run_id = manifest_run_id

    # Pre-bundle reports (may predate bundle creation, must share pipeline_run_id)
    pre_bundle_reports = {
        "generic_model_evaluation_report.json": "generic_evaluation",
        "model_quality_gate_report.json": "quality_gate",
    }

    # Post-bundle reports (must match bundle_id and be generated after bundle)
    post_bundle_reports = {
        "controlled_predicted_sql_execution_report.json": "controlled_predicted_sql",
        "model_selection_report.json": "model_selection",
    }

    eval_dir = bundle_path / "evaluation"

    for filename, name in pre_bundle_reports.items():
        report_path = eval_dir / filename
        if not report_path.exists():
            warnings.append(f"{name}: report file not found")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = validate_report_identity(
            report,
            expected_pipeline_run_id=expected_pipeline_run_id,
            expected_bundle_id=bundle_id,
            is_post_bundle=False,
            report_name=name,
        )
        report_results.append(result)
        issues.extend(result["issues"])

    for filename, name in post_bundle_reports.items():
        report_path = eval_dir / filename
        if not report_path.exists():
            warnings.append(f"{name}: report file not found")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = validate_report_identity(
            report,
            expected_pipeline_run_id=expected_pipeline_run_id,
            expected_bundle_id=bundle_id,
            is_post_bundle=True,
            report_name=name,
        )
        report_results.append(result)
        issues.extend(result["issues"])

        # Check timestamp ordering for post-bundle reports
        if manifest_created_at and result["generated_at"]:
            try:
                manifest_ts = _parse_timestamp(manifest_created_at)
                report_ts = _parse_timestamp(result["generated_at"])
                if report_ts < manifest_ts:
                    issues.append(
                        f"{name}: post-bundle report generated_at={result['generated_at']} "
                        f"is before bundle created_at={manifest_created_at}"
                    )
            except (TypeError, ValueError):
                warnings.append(f"{name}: could not parse timestamps for ordering check")

    # Check training report at the pipeline level
    pipeline_report_path = bundle_path / "pipeline" / "train_model_report.json"
    if pipeline_report_path.exists():
        report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
        result = validate_report_identity(
            report,
            expected_pipeline_run_id=expected_pipeline_run_id,
            is_post_bundle=False,
            report_name="train_model_report",
        )
        report_results.append(result)
        issues.extend(result["issues"])

    # Check dataset contribution report
    contrib_path = bundle_path / "generic_training" / "dataset_contribution_report.json"
    if contrib_path.exists():
        report = json.loads(contrib_path.read_text(encoding="utf-8"))
        result = validate_report_identity(
            report,
            expected_pipeline_run_id=expected_pipeline_run_id,
            is_post_bundle=False,
            report_name="dataset_contribution",
        )
        report_results.append(result)
        issues.extend(result["issues"])

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "reports": report_results,
        "expected_pipeline_run_id": expected_pipeline_run_id,
        "manifest_bundle_id": bundle_id,
    }


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp string to a datetime object."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
