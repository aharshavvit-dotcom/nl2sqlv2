from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedCandidate(BaseModel):
    rank: int
    example_id: str
    question: str
    dataset_name: str | None = None
    db_id: str | None = None
    intent: str | None = None
    template_id: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    sql_features: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float
    rerank_score: float | None = None
    schema_compatibility_score: float | None = None


class RuntimeSlot(BaseModel):
    slot_name: str
    value: str | int | None
    source: Literal["question", "retrieved_example", "sql_features", "default", "schema_match"]
    confidence: float
    alternatives: list[Any] = Field(default_factory=list)


class SchemaMapping(BaseModel):
    base_table: str | None = None
    metric_name: str | None = None
    metric_table: str | None = None
    metric_column: str | None = None
    metric_expression: str | None = None
    metric_aggregation: str | None = None
    metric_alias: str | None = None
    semantic_grain_risk: bool = False
    semantic_required_tables: list[str] = Field(default_factory=list)
    dimension_name: str | None = None
    dimension_table: str | None = None
    dimension_column: str | None = None
    entity_table: str | None = None
    date_table: str | None = None
    date_column: str | None = None
    filter_table: str | None = None
    filter_column: str | None = None
    filter_linking_method: str | None = None
    dimension_linking_method: str | None = None
    filter_ambiguous: bool = False
    filter_alternatives: list[str] = Field(default_factory=list)
    match_scores: dict[str, float] = Field(default_factory=dict)
    mapping_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JoinPlan(BaseModel):
    base_table: str
    required_tables: list[str] = Field(default_factory=list)
    join_clause: str = ""
    join_steps: list[dict[str, Any]] = Field(default_factory=list)
    join_policy: str | None = None
    confidence: float = 1.0
    warnings: list[str] = Field(default_factory=list)


class PredictionResult(BaseModel):
    question: str
    normalized_question: str
    status: Literal["completed", "abstained"] = "completed"
    source_model: Literal["generic_direct_planner", "retrieval_ir", "neural_ir", "adaptive_router",
                          # Backward-compatible values:
                          "option_c", "option_a", "hybrid"] = "retrieval_ir"
    intent: str | None = None
    template_id: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    schema_mapping: dict[str, Any] = Field(default_factory=dict)
    join_plan: dict[str, Any] | None = None
    query_ir: dict[str, Any] | None = None
    ir_validation: dict[str, Any] | None = None
    sql: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    conformal_threshold: float | None = None
    abstain: bool = False
    abstention_reason: str | None = None
    schema_drift_flags: list[str] = Field(default_factory=list)
    confidence_tier: str = "low"
    retrieved_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification: dict[str, Any] = Field(default_factory=dict)
    router_decision: dict[str, Any] = Field(default_factory=dict)
    neural_ir_version: str | None = None
    retrieval_ir_result: dict[str, Any] = Field(default_factory=dict)
    neural_ir_result: dict[str, Any] = Field(default_factory=dict)
    selected_query_ir: dict[str, Any] | None = None
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: dict[str, Any] = Field(default_factory=dict)
    filter_value_candidates: list[dict[str, Any]] = Field(default_factory=list)
    planner_debug: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)

    # ---- Backward-compatible aliases ----
    @property
    def option_a_version(self) -> str | None:
        """Deprecated alias. Use ``neural_ir_version``."""
        return self.neural_ir_version

    @property
    def option_a_result(self) -> dict[str, Any]:
        """Deprecated alias. Use ``neural_ir_result``."""
        return self.neural_ir_result

    @property
    def option_c_result(self) -> dict[str, Any]:
        """Deprecated alias. Use ``retrieval_ir_result``."""
        return self.retrieval_ir_result


def is_abstained_prediction(
    *,
    sql: str | None,
    prediction_status: str | None = None,
    requires_clarification: bool = False,
) -> bool:
    """Canonical abstention rule shared by runtime evaluation reports.

    Once SQL is emitted it is an answered prediction, even when confidence is
    low or validation later fails. Clarification only counts as abstention when
    no SQL was emitted.
    """
    if str(sql or "").strip():
        return False
    return prediction_status == "abstained" or bool(requires_clarification)
