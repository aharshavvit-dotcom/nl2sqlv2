"""Validate a model bundle against required structure and quality rules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .bundle_manifest import load_manifest


class ModelBundleValidator:
    """Validates that a model bundle directory is complete and safe."""

    _SENSITIVE_PATTERNS = re.compile(
        r"(password|secret|token|api_key|apikey|credential|connection_string|conn_str)",
        re.IGNORECASE,
    )

    def validate(self, bundle_dir: str | Path) -> dict[str, Any]:
        path = Path(bundle_dir)
        issues: list[str] = []
        warnings: list[str] = []
        checked: list[str] = []

        manifest_path = path / "bundle_manifest.json"
        checked.append(str(manifest_path))
        if not manifest_path.exists():
            issues.append("bundle_manifest.json not found")
            return _result(issues, warnings, checked)

        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            issues.append(f"Failed to parse bundle_manifest.json: {exc}")
            return _result(issues, warnings, checked)

        if manifest.status == "failed":
            issues.append("Bundle status is failed")

        manifest_data = manifest.to_dict()
        _check_no_secrets(manifest_data, issues)

        required_dirs = ["retrieval_ir", "evaluation", "generic_training", "configs"]
        if "neural_ir" in manifest.paths or manifest.artifacts.get("neural_manifest"):
            required_dirs.append("neural_ir")
        for key in required_dirs:
            rel = manifest.paths.get(key)
            if not rel:
                issues.append(f"Required bundle path missing from manifest: {key}")
                continue
            resolved = path / rel
            checked.append(str(resolved))
            if not resolved.exists():
                issues.append(f"Required artifact folder missing: {key} ({resolved})")

        retrieval_dir = path / manifest.paths.get("retrieval_ir", "retrieval_ir/")
        _require_files(
            retrieval_dir,
            ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"],
            "retrieval",
            issues,
            checked,
        )

        if "neural_ir" in required_dirs:
            neural_dir = path / manifest.paths.get("neural_ir", "neural_ir/")
            _require_files(neural_dir, ["model.pt", "config.yaml", "manifest.json"], "neural", issues, checked)

        eval_dir = path / manifest.paths.get("evaluation", "evaluation/")
        _require_files(eval_dir, ["generic_model_evaluation_report.json"], "evaluation", issues, checked)

        generic_dir = path / manifest.paths.get("generic_training", "generic_training/")
        contribution_path = generic_dir / "dataset_contribution_report.json"
        unsupported_path = generic_dir / "unsupported_sql_report.json"
        _require_files(generic_dir, ["dataset_contribution_report.json", "unsupported_sql_report.json"], "generic_training", issues, checked)
        if contribution_path.exists():
            contribution = _read_json(contribution_path)
            if not contribution.get("leakage_check_passed", False):
                issues.append("Dataset leakage check failed")
            requested = set(contribution.get("datasets_requested") or manifest.datasets or [])
            by_dataset = contribution.get("by_dataset") or {}
            for name in ["spider", "bird-mini"]:
                if name in requested and int((by_dataset.get(name) or {}).get("converted_to_queryir", 0)) <= 0:
                    issues.append(f"Requested dataset contributed zero usable examples: {name}")
        if unsupported_path.exists():
            checked.append(str(unsupported_path))

        qg = manifest.quality_gate or {}
        qg_required = bool(qg.get("required", False))
        qg_path = path / qg.get("report_path", "evaluation/model_quality_gate_report.json")
        checked.append(str(qg_path))
        if qg_required:
            if not qg_path.exists():
                issues.append("Required quality gate report missing")
            elif not _read_json(qg_path).get("passed", False):
                issues.append("Required quality gate failed")

        metrics = manifest.metrics or {}
        if float(metrics.get("unsafe_sql_count", 0) or 0) > 0:
            issues.append(f"Unsafe SQL count is {metrics.get('unsafe_sql_count')}, expected 0")
        if float(metrics.get("unnecessary_join_rate", 0.0) or 0.0) > 0.05:
            issues.append(f"Unnecessary join rate is {metrics.get('unnecessary_join_rate')}, max 0.05")
        if float(metrics.get("wrong_table_rate", 0.0) or 0.0) > 0.15:
            issues.append(f"Wrong table rate is {metrics.get('wrong_table_rate')}, max 0.15")
        sql_rate = metrics.get("sql_validation_rate")
        if isinstance(sql_rate, (int, float)) and sql_rate < 0:
            issues.append(f"SQL validation rate is invalid: {sql_rate}")

        return _result(issues, warnings, checked)


def _result(issues: list[str], warnings: list[str], checked: list[str]) -> dict[str, Any]:
    return {
        "passed": len(issues) == 0,
        "blocking_issues": issues,
        "warnings": warnings,
        "checked_files": checked,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _require_files(base: Path, names: list[str], label: str, issues: list[str], checked: list[str]) -> None:
    for name in names:
        target = base / name
        checked.append(str(target))
        if not target.exists():
            issues.append(f"Required {label} artifact missing: {target}")


def _check_no_secrets(data: Any, issues: list[str]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and ModelBundleValidator._SENSITIVE_PATTERNS.search(value):
                issues.append(f"Manifest contains sensitive-looking value at {key}")
            _check_no_secrets(value, issues)
    elif isinstance(data, list):
        for item in data:
            _check_no_secrets(item, issues)
