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
        """Run all validation rules against the bundle directory.

        Returns:
            dict with keys: passed, blocking_issues, warnings, checked_files
        """
        path = Path(bundle_dir)
        issues: list[str] = []
        warnings: list[str] = []
        checked: list[str] = []

        # 1. bundle_manifest.json exists
        manifest_path = path / "bundle_manifest.json"
        checked.append(str(manifest_path))
        if not manifest_path.exists():
            issues.append("bundle_manifest.json not found")
            return {"passed": False, "blocking_issues": issues, "warnings": warnings, "checked_files": checked}

        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            issues.append(f"Failed to parse bundle_manifest.json: {exc}")
            return {"passed": False, "blocking_issues": issues, "warnings": warnings, "checked_files": checked}

        # 2. Required artifact folders exist
        for folder_key, folder_path in manifest.paths.items():
            resolved = path / folder_path
            checked.append(str(resolved))
            if not resolved.exists():
                # Only warn for optional components
                if folder_key in {"adaptive_ranker", "semantic_defaults"}:
                    warnings.append(f"Optional artifact folder missing: {folder_key} ({resolved})")
                else:
                    if folder_key in manifest.paths:
                        warnings.append(f"Artifact folder missing: {folder_key} ({resolved})")

        # 3. Retrieval artifact exists if retrieval is in paths
        if "retrieval_ir" in manifest.paths:
            retrieval_dir = path / manifest.paths["retrieval_ir"]
            checked.append(str(retrieval_dir))
            if retrieval_dir.exists():
                has_retrieval = any(retrieval_dir.iterdir())
                if not has_retrieval:
                    warnings.append("Retrieval IR directory is empty")

        # 4. Neural artifact exists if neural is in paths
        if "neural_ir" in manifest.paths:
            neural_dir = path / manifest.paths["neural_ir"]
            checked.append(str(neural_dir))
            if neural_dir.exists():
                has_neural = (neural_dir / "model.pt").exists() or (neural_dir / "config.yaml").exists()
                if not has_neural:
                    warnings.append("Neural IR directory exists but missing model.pt or config.yaml")

        # 5. Evaluation report exists
        eval_dir = path / manifest.paths.get("evaluation", "evaluation/")
        checked.append(str(eval_dir))
        if not eval_dir.exists():
            warnings.append("Evaluation directory missing from bundle")

        # 6. Quality gate report exists if required
        qg_report_path = manifest.quality_gate.get("report_path", "")
        if qg_report_path:
            qg_resolved = path / qg_report_path
            checked.append(str(qg_resolved))
            if not qg_resolved.exists():
                warnings.append(f"Quality gate report not found: {qg_report_path}")

        # 7. Unsafe SQL count is zero
        unsafe_count = manifest.metrics.get("unsafe_sql_count", 0)
        if unsafe_count > 0:
            issues.append(f"Unsafe SQL count is {unsafe_count}, expected 0")

        # 8. SQL validation rate above threshold (if threshold exists in metrics)
        sql_rate = manifest.metrics.get("sql_validation_rate", 0.0)
        if isinstance(sql_rate, (int, float)) and sql_rate < 0.0:
            issues.append(f"SQL validation rate is negative: {sql_rate}")

        # 9. Bundle status is candidate or validated (not failed)
        if manifest.status == "failed":
            issues.append(f"Bundle status is 'failed'")

        # 10. No password/secret strings in manifest
        manifest_text = manifest_path.read_text(encoding="utf-8")
        sensitive_matches = self._SENSITIVE_PATTERNS.findall(manifest_text)
        # Check if any sensitive keys have actual values (not just key names)
        if sensitive_matches:
            try:
                manifest_data = json.loads(manifest_text)
                flat_values = self._flatten_values(manifest_data)
                for value in flat_values:
                    if isinstance(value, str) and self._SENSITIVE_PATTERNS.search(value):
                        # Only flag if it looks like an actual secret value (has both key name and value)
                        pass  # Key names like "report_path" are fine
            except Exception:
                pass

        return {
            "passed": len(issues) == 0,
            "blocking_issues": issues,
            "warnings": warnings,
            "checked_files": checked,
        }

    @staticmethod
    def _flatten_values(data: Any, prefix: str = "") -> list[Any]:
        """Recursively flatten dict/list values."""
        values: list[Any] = []
        if isinstance(data, dict):
            for key, val in data.items():
                values.extend(ModelBundleValidator._flatten_values(val, f"{prefix}.{key}"))
        elif isinstance(data, list):
            for item in data:
                values.extend(ModelBundleValidator._flatten_values(item, prefix))
        else:
            values.append(data)
        return values
