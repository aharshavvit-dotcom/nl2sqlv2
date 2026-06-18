"""Error Classifier — classifies prediction errors into actionable categories.

Takes comparison results from the GoldComparator and determines specific error
types so the self-improvement loop can generate targeted corrections and hard
negatives.
"""

from __future__ import annotations

import enum
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .gold_comparator import ComparisonResult


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

class ErrorCategory(str, enum.Enum):
    """Actionable error categories for prediction failures."""

    WRONG_INTENT = "wrong_intent"
    WRONG_BASE_TABLE = "wrong_base_table"
    WRONG_SELECTED_COLUMNS = "wrong_selected_columns"
    WRONG_METRIC = "wrong_metric"
    MISSING_METRIC = "missing_metric"
    WRONG_DIMENSION = "wrong_dimension"
    MISSING_DIMENSION = "missing_dimension"
    WRONG_FILTER = "wrong_filter"
    MISSING_FILTER = "missing_filter"
    EXTRA_FILTER = "extra_filter"
    WRONG_DATE_FILTER = "wrong_date_filter"
    MISSING_DATE_FILTER = "missing_date_filter"
    WRONG_JOIN = "wrong_join"
    UNNECESSARY_JOIN = "unnecessary_join"
    MISSING_JOIN = "missing_join"
    WRONG_GROUP_BY = "wrong_group_by"
    WRONG_ORDER_BY = "wrong_order_by"
    WRONG_ORDER = "wrong_order"
    WRONG_LIMIT = "wrong_limit"
    INVALID_SQL = "invalid_sql"
    UNSAFE_SQL = "unsafe_sql"
    EXECUTION_FAILED = "execution_failed"
    RESULT_MISMATCH = "result_mismatch"
    SQL_VALIDATION_FAILURE = "sql_validation_failure"
    IR_VALIDATION_FAILURE = "ir_validation_failure"
    CONVERSION_FAILURE = "conversion_failure"


# Map from QueryIR field name → error category
_FIELD_TO_CATEGORY: dict[str, ErrorCategory] = {
    "intent": ErrorCategory.WRONG_INTENT,
    "base_table": ErrorCategory.WRONG_BASE_TABLE,
    "metrics": ErrorCategory.WRONG_METRIC,
    "dimensions": ErrorCategory.WRONG_DIMENSION,
    "filters": ErrorCategory.WRONG_FILTER,
    "date_filters": ErrorCategory.WRONG_DATE_FILTER,
    "group_by": ErrorCategory.WRONG_GROUP_BY,
    "order_by": ErrorCategory.WRONG_ORDER,
    "limit": ErrorCategory.WRONG_LIMIT,
}

# Severity levels
_CRITICAL_CATEGORIES = {
    ErrorCategory.WRONG_INTENT,
    ErrorCategory.WRONG_BASE_TABLE,
    ErrorCategory.SQL_VALIDATION_FAILURE,
    ErrorCategory.INVALID_SQL,
    ErrorCategory.UNSAFE_SQL,
    ErrorCategory.EXECUTION_FAILED,
    ErrorCategory.IR_VALIDATION_FAILURE,
    ErrorCategory.CONVERSION_FAILURE,
}

_MAJOR_CATEGORIES = {
    ErrorCategory.WRONG_METRIC,
    ErrorCategory.WRONG_DIMENSION,
    ErrorCategory.WRONG_FILTER,
    ErrorCategory.MISSING_FILTER,
    ErrorCategory.EXTRA_FILTER,
    ErrorCategory.WRONG_JOIN,
    ErrorCategory.UNNECESSARY_JOIN,
    ErrorCategory.MISSING_JOIN,
    ErrorCategory.RESULT_MISMATCH,
}

# Suggested fix types per category
_FIX_TYPE: dict[ErrorCategory, str] = {
    ErrorCategory.WRONG_INTENT: "intent_correction",
    ErrorCategory.WRONG_BASE_TABLE: "table_correction",
    ErrorCategory.WRONG_SELECTED_COLUMNS: "column_correction",
    ErrorCategory.WRONG_METRIC: "column_correction",
    ErrorCategory.MISSING_METRIC: "column_correction",
    ErrorCategory.WRONG_DIMENSION: "column_correction",
    ErrorCategory.MISSING_DIMENSION: "column_correction",
    ErrorCategory.WRONG_FILTER: "filter_correction",
    ErrorCategory.MISSING_FILTER: "filter_correction",
    ErrorCategory.EXTRA_FILTER: "filter_correction",
    ErrorCategory.WRONG_DATE_FILTER: "filter_correction",
    ErrorCategory.MISSING_DATE_FILTER: "filter_correction",
    ErrorCategory.WRONG_JOIN: "join_correction",
    ErrorCategory.UNNECESSARY_JOIN: "join_correction",
    ErrorCategory.MISSING_JOIN: "join_correction",
    ErrorCategory.WRONG_GROUP_BY: "group_by_correction",
    ErrorCategory.WRONG_ORDER_BY: "order_correction",
    ErrorCategory.WRONG_ORDER: "order_correction",
    ErrorCategory.WRONG_LIMIT: "limit_correction",
    ErrorCategory.INVALID_SQL: "validation_fix",
    ErrorCategory.UNSAFE_SQL: "validation_fix",
    ErrorCategory.EXECUTION_FAILED: "execution_fix",
    ErrorCategory.RESULT_MISMATCH: "execution_fix",
    ErrorCategory.SQL_VALIDATION_FAILURE: "validation_fix",
    ErrorCategory.IR_VALIDATION_FAILURE: "validation_fix",
    ErrorCategory.CONVERSION_FAILURE: "conversion_fix",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ErrorClassification:
    """Classification result for a single prediction error."""

    example_id: str
    categories: list[ErrorCategory] = field(default_factory=list)
    severity: str = "minor"  # "critical" | "major" | "minor"
    details: dict[str, Any] = field(default_factory=dict)
    suggested_fix_type: str = ""


@dataclass
class ErrorReport:
    """Aggregated error report across a batch of predictions."""

    total_errors: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_dataset: dict[str, dict[str, int]] = field(default_factory=dict)
    top_error_categories: list[tuple[str, int]] = field(default_factory=list)
    classifications: list[ErrorClassification] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """Classifies prediction errors into actionable categories."""

    def classify(
        self,
        comparison: ComparisonResult,
        example: dict[str, Any],
    ) -> ErrorClassification:
        """Classify a single comparison result into error categories."""

        categories: list[ErrorCategory] = []
        details: dict[str, Any] = {}

        # Check QueryIR field-level mismatches
        for field_name, category in _FIELD_TO_CATEGORY.items():
            if not comparison.field_matches.get(field_name, True):
                categories.append(_refined_category(field_name, category, comparison.field_details.get(field_name, {})))
                details[category.value] = comparison.field_details.get(field_name, {})

        # Special join handling
        categories = self._classify_join_errors(comparison, categories, details)

        # Check validation failures from the example metadata
        if not _ir_valid(example):
            categories.append(ErrorCategory.IR_VALIDATION_FAILURE)
            details["ir_validation"] = example.get("ir_validation", {})

        if not _sql_valid(example):
            categories.append(ErrorCategory.SQL_VALIDATION_FAILURE)
            categories.append(ErrorCategory.INVALID_SQL)
            details["sql_validation"] = example.get("sql_validation", {})

        sql_validation = example.get("sql_validation") or example.get("validation") or {}
        issues_text = " ".join(str(item) for item in sql_validation.get("issues", []))
        if "blocked sql keyword" in issues_text.lower() or "unsafe" in issues_text.lower():
            categories.append(ErrorCategory.UNSAFE_SQL)

        execution_status = example.get("execution_status") or {}
        if execution_status and not bool(execution_status.get("success", execution_status.get("ok", False))):
            categories.append(ErrorCategory.EXECUTION_FAILED)
        if example.get("result_comparison") and not example["result_comparison"].get("result_match", False):
            categories.append(ErrorCategory.RESULT_MISMATCH)

        # Check for conversion failure
        pred_ir = example.get("predicted_query_ir") or example.get("query_ir")
        if pred_ir is None and example.get("prediction_failed"):
            categories.append(ErrorCategory.CONVERSION_FAILURE)

        # Deduplicate
        categories = list(dict.fromkeys(categories))

        severity = _severity(categories)

        # Determine suggested fix type (use the first/highest-severity category)
        fix_type = _FIX_TYPE.get(categories[0], "general_fix") if categories else ""

        return ErrorClassification(
            example_id=comparison.example_id,
            categories=categories,
            severity=severity,
            details=details,
            suggested_fix_type=fix_type,
        )

    def _classify_join_errors(
        self,
        comparison: ComparisonResult,
        categories: list[ErrorCategory],
        details: dict[str, Any],
    ) -> list[ErrorCategory]:
        """Refine join errors into unnecessary vs missing vs wrong."""

        if comparison.field_matches.get("joins", True):
            return categories

        join_details = comparison.field_details.get("joins", {})
        pred_joins = join_details.get("predicted")
        gold_joins = join_details.get("gold")
        pred_list = pred_joins if isinstance(pred_joins, list) else []
        gold_list = gold_joins if isinstance(gold_joins, list) else []

        # Remove generic WRONG_JOIN if we can be more specific
        refined = [c for c in categories if c != ErrorCategory.WRONG_JOIN]

        if pred_list and not gold_list:
            refined.append(ErrorCategory.UNNECESSARY_JOIN)
            details["unnecessary_join"] = {"predicted_join_count": len(pred_list)}
        elif gold_list and not pred_list:
            refined.append(ErrorCategory.MISSING_JOIN)
            details["missing_join"] = {"gold_join_count": len(gold_list)}
        else:
            refined.append(ErrorCategory.WRONG_JOIN)

        return refined

    def classify_batch(
        self,
        comparisons: list[ComparisonResult],
        examples: list[dict[str, Any]],
    ) -> ErrorReport:
        """Classify a batch of comparison results."""

        example_map = {str(ex.get("example_id", idx)): ex for idx, ex in enumerate(examples)}
        category_counter: Counter[str] = Counter()
        severity_counter: Counter[str] = Counter()
        dataset_counter: dict[str, Counter[str]] = defaultdict(Counter)
        classifications: list[ErrorClassification] = []

        for comp in comparisons:
            if comp.is_exact_match:
                continue  # No errors to classify

            example = example_map.get(comp.example_id, {})
            ec = self.classify(comp, example)

            if not ec.categories:
                continue

            classifications.append(ec)
            severity_counter[ec.severity] += 1
            dataset_name = example.get("dataset_name", "unknown")

            for cat in ec.categories:
                category_counter[cat.value] += 1
                dataset_counter[dataset_name][cat.value] += 1

        top_categories = category_counter.most_common()

        return ErrorReport(
            total_errors=len(classifications),
            by_category=dict(category_counter),
            by_severity=dict(severity_counter),
            by_dataset={name: dict(counts) for name, counts in dataset_counter.items()},
            top_error_categories=top_categories,
            classifications=classifications,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity(categories: list[ErrorCategory]) -> str:
    """Determine overall severity from a list of error categories."""
    if not categories:
        return "minor"
    if any(c in _CRITICAL_CATEGORIES for c in categories):
        return "critical"
    if any(c in _MAJOR_CATEGORIES for c in categories):
        return "major"
    return "minor"


def _refined_category(field_name: str, category: ErrorCategory, details: dict[str, Any]) -> ErrorCategory:
    predicted = details.get("predicted")
    gold = details.get("gold")
    pred_list = predicted if isinstance(predicted, list) else []
    gold_list = gold if isinstance(gold, list) else []
    if field_name == "metrics" and gold_list and not pred_list:
        return ErrorCategory.MISSING_METRIC
    if field_name == "dimensions" and gold_list and not pred_list:
        return ErrorCategory.MISSING_DIMENSION
    if field_name == "filters":
        if gold_list and not pred_list:
            return ErrorCategory.MISSING_FILTER
        if pred_list and not gold_list:
            return ErrorCategory.EXTRA_FILTER
    if field_name == "date_filters" and gold_list and not pred_list:
        return ErrorCategory.MISSING_DATE_FILTER
    if field_name == "order_by":
        return ErrorCategory.WRONG_ORDER_BY
    return category


def _ir_valid(example: dict[str, Any]) -> bool:
    """Check if the example's IR validation passed."""
    ir_val = example.get("ir_validation") or {}
    return bool(ir_val.get("is_valid", True))


def _sql_valid(example: dict[str, Any]) -> bool:
    """Check if the example's SQL validation passed."""
    sql_val = example.get("sql_validation") or example.get("validation") or {}
    return bool(sql_val.get("is_valid", sql_val.get("ok", True)))
