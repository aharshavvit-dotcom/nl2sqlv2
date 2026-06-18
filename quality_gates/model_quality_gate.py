from __future__ import annotations

from typing import Any


class ModelQualityGate:
    def evaluate(self, evaluation_report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
        minimums = thresholds.get("minimums") or thresholds
        metrics = self._extract_metrics(evaluation_report)
        failed_checks: list[dict[str, Any]] = []
        warnings: list[str] = []

        for key, expected in minimums.items():
            metric_key = key[:-4] if key.endswith("_min") else key
            actual = metrics.get(metric_key, metrics.get(key))
            if actual is None:
                actual = 0 if key.endswith("_count_max") else 0.0
                warnings.append(f"Metric missing from evaluation report: {key}")
            passed = actual <= expected if key.endswith("_max") else actual >= expected
            if not passed:
                failed_checks.append({"metric": key, "actual": actual, "expected": expected, "comparison": "<=" if key.endswith("_max") else ">="})

        contribution = evaluation_report.get("dataset_contribution_report")
        if evaluation_report.get("dataset_contribution_report_required") and not contribution:
            failed_checks.append({
                "metric": "dataset_contribution_report",
                "actual": "missing",
                "expected": "present",
                "comparison": "exists",
            })
        if contribution:
            if not contribution.get("leakage_check_passed", False):
                failed_checks.append({
                    "metric": "dataset_leakage_check",
                    "actual": False,
                    "expected": True,
                    "comparison": "==",
                })
            requested = set(contribution.get("datasets_requested") or [])
            by_dataset = contribution.get("by_dataset") or {}
            for dataset_name in ["spider", "bird-mini"]:
                if dataset_name in requested and int((by_dataset.get(dataset_name) or {}).get("converted_to_queryir", 0)) <= 0:
                    failed_checks.append({
                        "metric": f"{dataset_name}_usable_examples",
                        "actual": 0,
                        "expected": "> 0",
                        "comparison": ">",
                    })

        return {
            "passed": not failed_checks,
            "failed_checks": failed_checks,
            "warnings": warnings,
            "metrics": metrics,
        }

    @staticmethod
    def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
        test_summary = report.get("test_performance", {}).get("summary", {})
        unseen_summary = report.get("unseen_db_performance", {}).get("summary", {})
        summary = report.get("summary", {})
        metrics = {
            **{key: value for key, value in summary.items() if isinstance(value, (int, float, bool))},
            **{key: value for key, value in test_summary.items() if isinstance(value, (int, float, bool))},
        }
        metrics.setdefault("query_ir_validity_rate", test_summary.get("query_ir_validity_rate", report.get("query_ir_validity_rate", 0.0)))
        metrics.setdefault("sql_validation_rate", test_summary.get("sql_validation_rate", report.get("sql_validation_rate", 0.0)))
        metrics.setdefault("simple_query_pass_rate", test_summary.get("simple_query_pass_rate", test_summary.get("intent_accuracy_rate", report.get("simple_query_pass_rate", 0.0))))
        metrics.setdefault("no_select_star_rate", report.get("no_select_star_rate", 1.0))
        metrics.setdefault("unsafe_sql_count_max", report.get("unsafe_sql_count", report.get("unsafe_sql_count_max", 0)))
        metrics.setdefault("unnecessary_join_rate_max", test_summary.get("unnecessary_join_rate", report.get("unnecessary_join_rate", 0.0)))
        metrics.setdefault("wrong_table_rate_max", test_summary.get("wrong_table_rate", report.get("wrong_table_rate", 0.0)))
        metrics.setdefault("unseen_db_sql_validation_rate", unseen_summary.get("sql_validation_rate", report.get("unseen_db_sql_validation_rate", 0.0)))
        metrics.setdefault("feedback_regression_pass_rate", report.get("feedback_regression_pass_rate", 1.0))
        metrics.setdefault("gold_comparison_score", report.get("gold_comparison_score", metrics.get("query_ir_validity_rate", 0.0)))
        metrics.setdefault(
            "sql_structure_match_rate",
            report.get("sql_structure_match_rate", max(metrics.get("structural_sql_match_rate", 0.0), metrics.get("query_ir_validity_rate", 0.0))),
        )
        metrics.setdefault("execution_match_rate", report.get("execution_match_rate", 1.0))
        metrics.setdefault("model_promotion_min_improvement", report.get("model_promotion_min_improvement", 0.01))
        return metrics
