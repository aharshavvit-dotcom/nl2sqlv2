"""Centralized report schemas for evaluation, quality gate, and governance reports.

Provides typed Pydantic models to prevent metric-name drift across evaluation,
quality gates, bundle manifests, and route reports. Every report must include
a versioned schema identity.

Review Comment #4: Centralize report schemas and metric names.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


REPORT_SCHEMA_VERSION = "1.0"
METRIC_DEFINITIONS_VERSION = "2026.07.09"


class ReportIdentity(BaseModel):
    """Identity fields required on every report."""

    report_schema_version: str = REPORT_SCHEMA_VERSION
    report_type: str
    pipeline_run_id: str | None = None
    bundle_id: str | None = None
    generated_at: str | None = None
    metric_definitions_version: str = METRIC_DEFINITIONS_VERSION


class ApplicabilityAwareMetric(BaseModel):
    """A metric with applicability-aware denominator (Review Comment #6)."""

    value: float = 0.0
    numerator: int = 0
    denominator: int = 0
    applicable_cases: int = 0
    excluded_cases: int = 0

    @classmethod
    def compute(cls, matches: list[bool | None]) -> "ApplicabilityAwareMetric":
        """Compute metric, excluding None values from denominator."""
        applicable = [v for v in matches if v is not None]
        excluded = sum(1 for v in matches if v is None)
        numerator = sum(1 for v in applicable if v)
        denominator = len(applicable)
        return cls(
            value=numerator / denominator if denominator else 0.0,
            numerator=numerator,
            denominator=denominator,
            applicable_cases=denominator,
            excluded_cases=excluded,
        )


class SemanticPassResult(BaseModel):
    """Result of a single example's strict semantic pass check."""

    passed: bool = False
    applicable_checks: list[str] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    primary_failure_reason: str | None = None


class SemanticEvaluationMetrics(BaseModel):
    """Semantic correctness metrics with applicability-aware denominators."""

    # Safety / validity pass rates (all queries in denominator)
    simple_query_safety_pass_rate: float = 0.0
    simple_query_validity_pass_rate: float = 0.0
    simple_query_table_pass_rate: float = 0.0

    # Semantic pass rates (applicability-aware)
    simple_query_projection_pass_rate: float = 0.0
    simple_query_filter_pass_rate: float = 0.0
    simple_query_semantic_pass_rate: float = 0.0

    # Detailed applicability-aware metrics
    projection_exact_match: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    projection_contains_gold: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    extra_projection_column: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    filter_column_accuracy: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    filter_value_accuracy: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    filter_value_extraction: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    filter_column_top1: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    filter_column_top3: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)
    dimension_column_accuracy: ApplicabilityAwareMetric = Field(default_factory=ApplicabilityAwareMetric)

    # Minimum support requirements for production gating
    minimum_applicable_cases: int = 50


class RouteEvaluationMetrics(BaseModel):
    """Per-route semantic metrics."""

    route_distribution: dict[str, int] = Field(default_factory=dict)
    route_percentage: dict[str, float] = Field(default_factory=dict)
    semantic_accuracy_by_route: dict[str, float] = Field(default_factory=dict)
    filter_accuracy_by_route: dict[str, float] = Field(default_factory=dict)
    projection_accuracy_by_route: dict[str, float] = Field(default_factory=dict)
    latency_by_route: dict[str, float] = Field(default_factory=dict)


class RowAccounting(BaseModel):
    """Explicit row accounting (prevents hidden denominator mismatches)."""

    standard_rows_evaluated: int = 0
    unseen_db_rows_evaluated: int = 0
    total_rows_evaluated: int = 0
    standard_predictions_generated: int = 0
    unseen_db_predictions_generated: int = 0
    total_predictions_generated: int = 0


class PromotionEligibility(BaseModel):
    """Promotion eligibility decision with explicit blockers.

    The evaluator describes evidence; quality gate checks thresholds;
    centralized promotion policy makes the final decision (Review #5).
    """

    eligible_for_promotion: bool = False
    evaluation_scope: str = "unknown"
    full_bundle_runtime_used: bool = False
    report_identity_strength: Literal["strong", "weak", "missing"] = "missing"
    valid_as_production_evidence: bool = False
    promotion_blockers: list[str] = Field(default_factory=list)


class CheckpointIdentity(BaseModel):
    """Unambiguous checkpoint identity (Review #16)."""

    selected_checkpoint_file: str | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_sha256: str | None = None
    runtime_export_file: str | None = None
    runtime_export_source_sha256: str | None = None
    runtime_export_equivalent_to_selected_checkpoint: bool | None = None


# --- Failure categories for QueryIR semantic diff ---

FAILURE_CATEGORIES = [
    "intent_mismatch",
    "base_table_mismatch",
    "projection_missing_column",
    "projection_extra_column",
    "projection_mismatch",
    "metric_column_mismatch",
    "dimension_column_mismatch",
    "filter_column_mismatch",
    "filter_operator_mismatch",
    "filter_value_mismatch",
    "date_filter_mismatch",
    "aggregation_mismatch",
    "group_by_mismatch",
    "join_mismatch",
    "order_by_mismatch",
    "limit_mismatch",
    "row_count_mismatch",
    "result_value_mismatch",
]


# --- Staged quality gate thresholds ---

DEVELOPMENT_GATE = {
    "projection_exact_match_rate": 0.50,
    "filter_column_accuracy_rate": 0.55,
    "filter_value_accuracy_rate": 0.45,
    "dimension_column_accuracy_rate": 0.50,
    "safe_but_wrong_sql_rate_max": 0.50,
}

RC_GATE = {
    "projection_exact_match_rate": 0.65,
    "filter_column_accuracy_rate": 0.65,
    "filter_value_accuracy_rate": 0.60,
    "dimension_column_accuracy_rate": 0.60,
    "controlled_execution_match_rate": 0.65,
    "safe_but_wrong_sql_rate_max": 0.35,
}

PRODUCTION_GATE = {
    "projection_exact_match_rate": 0.70,
    "filter_column_accuracy_rate": 0.70,
    "filter_value_accuracy_rate": 0.70,
    "filter_value_extraction_rate": 0.80,
    "dimension_column_accuracy_rate": 0.65,
    "controlled_execution_match_rate": 0.70,
    "result_value_match_rate": 0.70,
    "safe_but_wrong_sql_rate_max": 0.30,
    "unsafe_sql_count_max": 0,
}


# --- Stage 2: Failure Attribution & Route Diagnostics schemas (Revisions #11, #15, #16) ---

class RoutePredictionResult(BaseModel):
    route: str
    available: bool = True
    unavailable_reason: str | None = None
    native_query_ir: dict[str, Any] | None = None
    resolved_query_ir: dict[str, Any] | None = None
    validated_query_ir: dict[str, Any] | None = None
    rendered_sql: str | None = None
    sql_validation: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    semantic_pass: bool = False
    failure_stage: str | None = None


class RouteDiagnosticCase(BaseModel):
    example_id: str
    question: str
    dataset: str
    complexity: str
    intent: str | None = None
    route_results: dict[str, RoutePredictionResult] = Field(default_factory=dict)
    selected_route: str
    passing_routes: list[str] = Field(default_factory=list)
    selected_route_passed: bool = False
    oracle_route_available: bool = False
    router_regret: bool = False


class RendererAttributionResult(BaseModel):
    example_id: str
    question: str
    predicted_ir_correct: bool = False
    gold_ir_render_success: bool = False
    gold_ir_rendered_sql: str | None = None
    gold_ir_sql_validation: dict[str, Any] | None = None
    predicted_ir_render_success: bool = False
    predicted_ir_rendered_sql: str | None = None
    predicted_ir_sql_validation: dict[str, Any] | None = None
    meaning: str
    failure_stage: str


class RouteDiagnosticReport(BaseModel):
    report_schema_version: str = REPORT_SCHEMA_VERSION
    report_type: Literal["route_diagnostics"] = "route_diagnostics"
    pipeline_run_id: str | None = None
    generated_at: str | None = None
    dataset_hash: str | None = None
    schema_hash: str | None = None
    retrieval_artifact_hash: str | None = None
    neural_checkpoint_hash: str | None = None
    routing_policy_hash: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    by_dataset: dict[str, Any] = Field(default_factory=dict)
    by_intent: dict[str, Any] = Field(default_factory=dict)
    by_complexity: dict[str, Any] = Field(default_factory=dict)

