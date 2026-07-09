"""Strict semantic pass calculation for NL-to-SQL evaluation.

A query passes semantically only when ALL applicable slots match the gold QueryIR.
This replaces the weak simple_query_pass that only checked intent, base_table,
joins and SQL validity.

Review requirement: Do not call broad validity success "simple-query pass."
"""

from __future__ import annotations

from typing import Any

from ir.query_ir_models import diff_query_ir
from evaluation.report_schemas import (
    ApplicabilityAwareMetric,
    SemanticEvaluationMetrics,
    SemanticPassResult,
)


# Checks that apply to every query
_UNIVERSAL_CHECKS = ["sql_safety", "sql_validity"]

# Checks that apply based on gold IR content
_CONDITIONAL_CHECKS = {
    "intent": lambda gold: gold.get("intent") is not None,
    "base_table": lambda gold: gold.get("base_table") is not None,
    "projection": lambda gold: bool(gold.get("dimensions") or gold.get("metrics")),
    "metric": lambda gold: bool(gold.get("metrics")),
    "dimension": lambda gold: bool(gold.get("dimensions")),
    "filter_column": lambda gold: bool(gold.get("filters")),
    "filter_operator": lambda gold: bool(gold.get("filters")),
    "filter_value": lambda gold: bool(gold.get("filters")),
    "aggregation": lambda gold: bool(gold.get("metrics")),
    "group_by": lambda gold: bool(gold.get("group_by")),
    "order_by": lambda gold: bool(gold.get("order_by")),
    "limit": lambda gold: gold.get("limit") is not None and gold.get("limit") != 100,
    "join": lambda gold: bool(gold.get("joins")),
    "date_filter": lambda gold: bool(gold.get("date_filters")),
}


def compute_simple_query_semantic_pass(
    gold_ir: dict[str, Any] | None,
    predicted_ir: dict[str, Any] | None,
    final_sql: str | None,
    validation_result: dict[str, Any] | None,
) -> SemanticPassResult:
    """Compute strict semantic pass for a single query.

    A query passes only when ALL applicable checks succeed.
    """
    gold = gold_ir or {}
    predicted = predicted_ir or {}
    validation = validation_result or {}

    # Get canonical diff
    diff = diff_query_ir(predicted, gold)

    applicable_checks: list[str] = []
    passed_checks: list[str] = []
    failed_checks: list[str] = []

    # Universal checks
    sql_safe = _check_sql_safety(final_sql, validation)
    _record_check("sql_safety", sql_safe, applicable_checks, passed_checks, failed_checks)

    sql_valid = bool(validation.get("is_valid", True)) and bool(final_sql)
    _record_check("sql_validity", sql_valid, applicable_checks, passed_checks, failed_checks)

    # Diff-based checks
    _diff_check_map = {
        "intent": "intent_match",
        "base_table": "base_table_match",
        "projection": "projection_match",
        "metric": "metric_match",
        "dimension": "dimension_match",
        "filter_column": "filter_column_match",
        "filter_operator": "filter_operator_match",
        "filter_value": "filter_value_match",
        "aggregation": "aggregation_match",
        "group_by": "group_by_match",
        "order_by": "order_by_match",
        "limit": "limit_match",
        "join": "join_match",
        "date_filter": "date_filter_match",
    }

    for check_name, condition_fn in _CONDITIONAL_CHECKS.items():
        if not condition_fn(gold):
            continue
        diff_key = _diff_check_map[check_name]
        check_passed = diff.get(diff_key, True) is True
        _record_check(check_name, check_passed, applicable_checks, passed_checks, failed_checks)

    # Determine primary failure
    primary_failure = failed_checks[0] if failed_checks else None

    return SemanticPassResult(
        passed=len(failed_checks) == 0,
        applicable_checks=applicable_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        primary_failure_reason=primary_failure,
    )


def compute_semantic_evaluation_metrics(
    per_example: list[dict[str, Any]],
) -> SemanticEvaluationMetrics:
    """Compute all semantic evaluation metrics with applicability-aware denominators.

    Args:
        per_example: List of per-example evaluation results, each containing
            'semantic_pass', 'filter_linking', 'projection', and 'dimension_linking'.
    """
    total = len(per_example) or 1

    # Safety / validity (all queries in denominator)
    safety_passes = sum(
        1 for item in per_example
        if _check_sql_safety(
            item.get("final_sql_after_repair") or item.get("predicted_sql"),
            item.get("sql_validation") or {},
        )
    )
    validity_passes = sum(
        1 for item in per_example
        if item.get("sql_validation_passed", False)
    )
    table_passes = sum(
        1 for item in per_example
        if item.get("base_table_correct", False)
    )

    # Semantic passes
    semantic_pass_results = [item.get("semantic_pass") or {} for item in per_example]
    projection_pass_count = sum(
        1 for sp in semantic_pass_results
        if "projection" not in (sp.get("failed_checks") or [])
        and "projection" in (sp.get("applicable_checks") or [])
    )
    projection_applicable = sum(
        1 for sp in semantic_pass_results
        if "projection" in (sp.get("applicable_checks") or [])
    )
    filter_pass_count = sum(
        1 for sp in semantic_pass_results
        if all(
            check not in (sp.get("failed_checks") or [])
            for check in ("filter_column", "filter_value", "filter_operator")
        )
        and any(
            check in (sp.get("applicable_checks") or [])
            for check in ("filter_column", "filter_value", "filter_operator")
        )
    )
    filter_applicable = sum(
        1 for sp in semantic_pass_results
        if any(
            check in (sp.get("applicable_checks") or [])
            for check in ("filter_column", "filter_value", "filter_operator")
        )
    )
    full_pass_count = sum(
        1 for sp in semantic_pass_results
        if sp.get("passed", False)
    )

    # Applicability-aware linking metrics
    filter_col_matches = [
        _get_linking_bool(item, "filter_linking", "filter_column_match")
        for item in per_example
    ]
    filter_val_matches = [
        _get_linking_bool(item, "filter_linking", "filter_value_match")
        for item in per_example
    ]
    filter_val_extract_matches = [
        _get_linking_bool(item, "filter_linking", "filter_value_extraction_match")
        for item in per_example
    ]
    filter_col_top1_matches = [
        _get_linking_bool(item, "filter_linking", "filter_column_top1_match")
        for item in per_example
    ]
    filter_col_top3_matches = [
        _get_linking_bool(item, "filter_linking", "filter_column_top3_match")
        for item in per_example
    ]
    dimension_matches = [
        _get_linking_bool(item, "dimension_linking", "dimension_match")
        for item in per_example
    ]
    projection_exact = [
        _get_projection_bool(item, "exact_match")
        for item in per_example
    ]
    projection_contains = [
        _get_projection_bool(item, "contains_gold")
        for item in per_example
    ]
    projection_extra = [
        _get_projection_bool(item, "has_extra_columns")
        for item in per_example
    ]

    return SemanticEvaluationMetrics(
        simple_query_safety_pass_rate=safety_passes / total,
        simple_query_validity_pass_rate=validity_passes / total,
        simple_query_table_pass_rate=table_passes / total,
        simple_query_projection_pass_rate=(
            projection_pass_count / projection_applicable if projection_applicable else 0.0
        ),
        simple_query_filter_pass_rate=(
            filter_pass_count / filter_applicable if filter_applicable else 0.0
        ),
        simple_query_semantic_pass_rate=full_pass_count / total,
        projection_exact_match=ApplicabilityAwareMetric.compute(projection_exact),
        projection_contains_gold=ApplicabilityAwareMetric.compute(projection_contains),
        extra_projection_column=ApplicabilityAwareMetric.compute(projection_extra),
        filter_column_accuracy=ApplicabilityAwareMetric.compute(filter_col_matches),
        filter_value_accuracy=ApplicabilityAwareMetric.compute(filter_val_matches),
        filter_value_extraction=ApplicabilityAwareMetric.compute(filter_val_extract_matches),
        filter_column_top1=ApplicabilityAwareMetric.compute(filter_col_top1_matches),
        filter_column_top3=ApplicabilityAwareMetric.compute(filter_col_top3_matches),
        dimension_column_accuracy=ApplicabilityAwareMetric.compute(dimension_matches),
    )


def _record_check(
    name: str,
    passed: bool,
    applicable: list[str],
    passed_list: list[str],
    failed_list: list[str],
) -> None:
    applicable.append(name)
    if passed:
        passed_list.append(name)
    else:
        failed_list.append(name)


def _check_sql_safety(sql: str | None, validation: dict[str, Any]) -> bool:
    """Check SQL safety (no unsafe statements)."""
    if not sql:
        return True  # No SQL = safe (abstention)
    normalized = sql.strip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False
    if validation.get("is_safe") is False:
        return False
    return True


def _get_linking_bool(
    item: dict[str, Any],
    section: str,
    key: str,
) -> bool | None:
    """Extract a linking diagnostic boolean, returning None if not applicable."""
    linking = item.get(section)
    if not isinstance(linking, dict):
        return None
    value = linking.get(key)
    if value is None:
        return None
    return bool(value)


def _get_projection_bool(
    item: dict[str, Any],
    key: str,
) -> bool | None:
    """Extract a projection diagnostic boolean, returning None if not applicable."""
    projection = item.get("projection")
    if not isinstance(projection, dict):
        return None
    # Only applicable if there are gold columns or predicted columns
    gold_cols = projection.get("gold_columns") or []
    if not gold_cols:
        return None
    value = projection.get(key)
    if value is None:
        return None
    return bool(value)
