from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .taxonomy import Capability, SafetyLabel, SUPPORTED_QUERYIR_V1_CAPABILITIES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class JoinEdge(StrictModel):
    left_table: str | None = None
    left_column: str | None = None
    right_table: str | None = None
    right_column: str | None = None
    condition: str | None = None
    join_type: str | None = None


class CorrelatedSubqueryInfo(StrictModel):
    outer_scope_tables: list[str] = Field(default_factory=list)
    inner_scope_tables: list[str] = Field(default_factory=list)
    correlated_columns: list[str] = Field(default_factory=list)
    correlation_operators: list[str] = Field(default_factory=list)


class WindowFunctionInfo(StrictModel):
    function: str
    arguments: list[str] = Field(default_factory=list)
    partition_columns: list[str] = Field(default_factory=list)
    order_columns: list[str] = Field(default_factory=list)
    order_directions: list[str] = Field(default_factory=list)
    frame_definition: str | None = None


class SetOperationBranch(StrictModel):
    branch_index: int
    sql: str
    required_capabilities: list[Capability] = Field(default_factory=list)


class TaskMasks(StrictModel):
    capability: int = 0
    safety: int = 0
    table: int = 0
    column: int = 0
    aggregation: int = 0
    filter: int = 0
    join_edge: int = 0
    complexity: int = 0
    contrastive_schema_linking: int = 0
    subquery: int = 0
    window: int = 0
    set_operation: int = 0
    full_query_ir: int = 0


class PartialSQLSupervision(StrictModel):
    required_capabilities: list[Capability] = Field(default_factory=list)
    safety_labels: list[SafetyLabel] = Field(default_factory=list)
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)
    selected_columns: list[str] = Field(default_factory=list)
    aggregation_functions: list[str] = Field(default_factory=list)
    group_by_columns: list[str] = Field(default_factory=list)
    filter_columns: list[str] = Field(default_factory=list)
    filter_operators: list[str] = Field(default_factory=list)
    join_edges: list[JoinEdge] = Field(default_factory=list)
    join_path_length: int | None = None
    subquery_types: list[str] = Field(default_factory=list)
    subquery_depth: int = 0
    correlated_subqueries: list[CorrelatedSubqueryInfo] = Field(default_factory=list)
    window_functions: list[WindowFunctionInfo] = Field(default_factory=list)
    window_partition_columns: list[str] = Field(default_factory=list)
    window_order_columns: list[str] = Field(default_factory=list)
    set_operation: str | None = None
    set_operation_branches: list[SetOperationBranch] = Field(default_factory=list)
    has_case: bool = False
    has_having: bool = False
    full_query_ir_supported: bool = False
    unsupported_reason: str | None = None
    extraction_status: str = "ok"
    validation_errors: list[str] = Field(default_factory=list)

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def _unique_capabilities(cls, value: Any) -> list[Any]:
        return _dedupe(value)

    @field_validator("safety_labels", mode="before")
    @classmethod
    def _unique_safety_labels(cls, value: Any) -> list[Any]:
        return _dedupe(value)


class CapabilityAnnotation(StrictModel):
    example_id: str
    dataset_source: str
    database_identifier: str
    sql_dialect: str = "sqlite"
    parser_version: str
    annotation_version: str = "capability_taxonomy_v1"
    schema_fingerprint: str = "unknown"
    extraction_status: str = "ok"
    validation_errors: list[str] = Field(default_factory=list)
    understood: bool = False
    required_capabilities: list[Capability] = Field(default_factory=list)
    supported_capabilities: list[Capability] = Field(default_factory=lambda: sorted(SUPPORTED_QUERYIR_V1_CAPABILITIES, key=lambda item: item.value))
    currently_supported: bool = False
    unsupported_required_capabilities: list[Capability] = Field(default_factory=list)
    safety_labels: list[SafetyLabel] = Field(default_factory=list)
    partial_supervision: PartialSQLSupervision
    task_masks: TaskMasks = Field(default_factory=TaskMasks)

    @field_validator("required_capabilities", "supported_capabilities", "unsupported_required_capabilities", mode="before")
    @classmethod
    def _unique_capability_lists(cls, value: Any) -> list[Any]:
        return _dedupe(value)


class SupportedQueryIRExample(StrictModel):
    example_id: str
    dataset_source: str
    database_identifier: str
    sql_dialect: str = "sqlite"
    parser_version: str
    annotation_version: str = "capability_taxonomy_v1"
    schema_fingerprint: str
    extraction_status: str
    validation_errors: list[str] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    safety_labels: list[SafetyLabel] = Field(default_factory=list)
    partial_supervision: PartialSQLSupervision
    query_ir: dict[str, Any]
    task_masks: TaskMasks


class UnsupportedExecutableExample(StrictModel):
    example_id: str
    dataset_source: str
    database_identifier: str
    sql_dialect: str = "sqlite"
    parser_version: str
    annotation_version: str = "capability_taxonomy_v1"
    schema_fingerprint: str
    extraction_status: str
    validation_errors: list[str] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    safety_labels: list[SafetyLabel] = Field(default_factory=list)
    partial_supervision: PartialSQLSupervision
    unsupported_reason: str
    task_masks: TaskMasks


class SafetyExample(StrictModel):
    example_id: str
    dataset_source: str
    database_identifier: str
    sql_dialect: str = "sqlite"
    parser_version: str
    annotation_version: str = "capability_taxonomy_v1"
    schema_fingerprint: str
    extraction_status: str
    validation_errors: list[str] = Field(default_factory=list)
    safety_labels: list[SafetyLabel]
    partial_supervision: PartialSQLSupervision
    task_masks: TaskMasks


def annotation_to_supported_example(annotation: CapabilityAnnotation, query_ir: dict[str, Any]) -> SupportedQueryIRExample:
    return SupportedQueryIRExample(
        example_id=annotation.example_id,
        dataset_source=annotation.dataset_source,
        database_identifier=annotation.database_identifier,
        sql_dialect=annotation.sql_dialect,
        parser_version=annotation.parser_version,
        annotation_version=annotation.annotation_version,
        schema_fingerprint=annotation.schema_fingerprint,
        extraction_status=annotation.extraction_status,
        validation_errors=annotation.validation_errors,
        capabilities=annotation.required_capabilities,
        safety_labels=annotation.safety_labels,
        partial_supervision=annotation.partial_supervision,
        query_ir=query_ir,
        task_masks=annotation.task_masks,
    )


def annotation_to_unsupported_example(annotation: CapabilityAnnotation, unsupported_reason: str) -> UnsupportedExecutableExample:
    return UnsupportedExecutableExample(
        example_id=annotation.example_id,
        dataset_source=annotation.dataset_source,
        database_identifier=annotation.database_identifier,
        sql_dialect=annotation.sql_dialect,
        parser_version=annotation.parser_version,
        annotation_version=annotation.annotation_version,
        schema_fingerprint=annotation.schema_fingerprint,
        extraction_status=annotation.extraction_status,
        validation_errors=annotation.validation_errors,
        capabilities=annotation.required_capabilities,
        safety_labels=annotation.safety_labels,
        partial_supervision=annotation.partial_supervision,
        unsupported_reason=unsupported_reason,
        task_masks=annotation.task_masks,
    )


def _dedupe(value: Any) -> list[Any]:
    if value is None:
        return []
    seen: set[str] = set()
    result: list[Any] = []
    for item in value:
        key = item.value if hasattr(item, "value") else str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
