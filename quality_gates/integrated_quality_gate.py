"""Integrated quality gate for the training pipeline.

Validates that a training run meets all quality thresholds
before allowing model bundle promotion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_quality_gate import ModelQualityGate
from .thresholds import load_thresholds


class IntegratedQualityGate:
    """Comprehensive quality gate that checks all required metrics."""

    def evaluate(
        self,
        evaluation_report: dict[str, Any] | None = None,
        quality_gate_report: dict[str, Any] | None = None,
        bundle_validation: dict[str, Any] | None = None,
        thresholds_path: str | Path = "evaluation/model_quality_thresholds.yaml",
    ) -> dict[str, Any]:
        """Run all integrated quality checks.

        Validates:
            1. Dataset leakage report passed.
            2. QueryIR validation rate meets threshold.
            3. SQL validation rate meets threshold.
            4. Unsafe SQL count is zero.
            5. No SELECT * rate is 100%.
            6. Unnecessary join rate below threshold.
            7. Wrong table rate below threshold.
            8. Simple direct query pass rate meets threshold.
            9. Evaluation report exists.
            10. Bundle validation passed.

        Returns:
            dict with passed, failed_checks, warnings, metrics
        """
        thresholds = load_thresholds(thresholds_path)
        failed_checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # Use existing ModelQualityGate for metric-based checks
        if evaluation_report:
            gate = ModelQualityGate()
            gate_result = gate.evaluate(evaluation_report, thresholds)
            failed_checks.extend(gate_result.get("failed_checks", []))
            warnings.extend(gate_result.get("warnings", []))
            metrics.update(gate_result.get("metrics", {}))
        else:
            warnings.append("No evaluation report available for quality gate")

        # 1. Dataset leakage check
        summary = (evaluation_report or {}).get("summary", {})
        leakage = summary.get("dataset_leakage_passed")
        if leakage is not None and not leakage:
            failed_checks.append({
                "metric": "dataset_leakage",
                "actual": False,
                "expected": True,
                "comparison": "==",
            })

        # 4. Unsafe SQL count
        unsafe_count = summary.get("unsafe_sql_count", metrics.get("unsafe_sql_count_max", 0))
        if isinstance(unsafe_count, (int, float)) and unsafe_count > 0:
            if not any(c.get("metric") == "unsafe_sql_count_max" for c in failed_checks):
                failed_checks.append({
                    "metric": "unsafe_sql_count",
                    "actual": unsafe_count,
                    "expected": 0,
                    "comparison": "<=",
                })

        # 9. Evaluation report exists check
        if not evaluation_report:
            failed_checks.append({
                "metric": "evaluation_report_exists",
                "actual": False,
                "expected": True,
                "comparison": "==",
            })

        # 10. Bundle validation passed
        if bundle_validation is not None and not bundle_validation.get("passed", False):
            failed_checks.append({
                "metric": "bundle_validation",
                "actual": False,
                "expected": True,
                "comparison": "==",
            })
            for issue in bundle_validation.get("blocking_issues", []):
                warnings.append(f"Bundle: {issue}")

        passed = len(failed_checks) == 0
        return {
            "passed": passed,
            "failed_checks": failed_checks,
            "warnings": warnings,
            "metrics": metrics,
        }

    def evaluate_and_write(
        self,
        output_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run quality gate and write report to file."""
        result = self.evaluate(**kwargs)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
