from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QUERY_IR_V2_VERSION = "2.0"


class QueryIRV2Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConfidenceMetadata(QueryIRV2Base):
    overall: float | None = None
    slots: dict[str, float] = Field(default_factory=dict)
    source: str | None = None

    @field_validator("overall")
    @classmethod
    def _overall_in_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("overall confidence must be between 0.0 and 1.0")
        return value

    @field_validator("slots")
    @classmethod
    def _slot_confidence_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        for slot, score in value.items():
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"confidence for slot {slot!r} must be between 0.0 and 1.0")
        return value


class CapabilityMetadata(QueryIRV2Base):
    required_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list[str] = Field(default_factory=list)
    renderer_capabilities: list[str] = Field(default_factory=list)
    source_capability_labels: list[str] = Field(default_factory=list)


class ColumnExpression(QueryIRV2Base):
    expression_type: Literal["COLUMN"] = "COLUMN"
    table: str | None = None
    column: str
    raw_expression: str | None = None

    @field_validator("table", "column")
    @classmethod
    def _identifier_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not str(value).strip():
            raise ValueError("identifier values must not be blank")
        return value


class LiteralValueType(str, Enum):
    """SQL-aware literal value types with precision-preserving semantics.

    DECIMAL values are stored as canonical strings to preserve scale and precision.
    DATE/TIME/TIMESTAMP values use ISO format strings.
    """
    NULL = "NULL"
    STRING = "STRING"
    INTEGER = "INTEGER"
    DECIMAL = "DECIMAL"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMP_WITH_TIMEZONE = "TIMESTAMP_WITH_TIMEZONE"
    BINARY = "BINARY"


# Strict serializable union — no arbitrary Python objects.
# DECIMAL, DATE, TIME, TIMESTAMP values are canonical strings.
LiteralValue: TypeAlias = str | int | bool | None


class LiteralExpression(QueryIRV2Base):
    expression_type: Literal["LITERAL"] = "LITERAL"
    value: LiteralValue
    value_type: LiteralValueType = LiteralValueType.STRING
    source_text: str | None = None  # preserves original textual representation

    @field_validator("value_type", mode="before")
    @classmethod
    def _coerce_value_type(cls, v: Any) -> LiteralValueType:
        """Accept legacy lowercase strings for backward compatibility."""
        if isinstance(v, str) and v not in LiteralValueType.__members__:
            mapping = {
                "string": LiteralValueType.STRING,
                "integer": LiteralValueType.INTEGER,
                "float": LiteralValueType.DECIMAL,
                "decimal": LiteralValueType.DECIMAL,
                "boolean": LiteralValueType.BOOLEAN,
                "null": LiteralValueType.NULL,
                "date": LiteralValueType.DATE,
                "time": LiteralValueType.TIME,
                "timestamp": LiteralValueType.TIMESTAMP,
            }
            return mapping.get(v.lower(), LiteralValueType.STRING)
        return v


class FunctionExpression(QueryIRV2Base):
    expression_type: Literal["FUNCTION"] = "FUNCTION"
    name: str
    arguments: list[Expression] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("function name must not be blank")
        return value


class AggregationExpression(QueryIRV2Base):
    expression_type: Literal["AGGREGATION"] = "AGGREGATION"
    function: str
    argument: Expression | None = None
    distinct: bool = False

    @field_validator("function")
    @classmethod
    def _function_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("aggregation function must not be blank")
        return value.upper()


class BinaryOperationExpression(QueryIRV2Base):
    expression_type: Literal["BINARY_OPERATION"] = "BINARY_OPERATION"
    operator: str
    left: Expression
    right: Expression

    @field_validator("operator")
    @classmethod
    def _operator_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("binary operator must not be blank")
        return value


class UnaryOperationExpression(QueryIRV2Base):
    expression_type: Literal["UNARY_OPERATION"] = "UNARY_OPERATION"
    operator: str
    operand: Expression


class BooleanOperationExpression(QueryIRV2Base):
    expression_type: Literal["BOOLEAN_OPERATION"] = "BOOLEAN_OPERATION"
    operator: Literal["AND", "OR", "NOT"]
    operands: list[Expression]

    @model_validator(mode="after")
    def _validate_operand_count(self) -> "BooleanOperationExpression":
        minimum = 1 if self.operator == "NOT" else 2
        if len(self.operands) < minimum:
            raise ValueError(f"{self.operator} requires at least {minimum} operand(s)")
        return self


class CaseWhen(QueryIRV2Base):
    when: Predicate
    then: Expression


class CaseExpression(QueryIRV2Base):
    expression_type: Literal["CASE_EXPRESSION"] = "CASE_EXPRESSION"
    cases: list[CaseWhen]
    else_expression: Expression | None = None


class SubqueryExpression(QueryIRV2Base):
    expression_type: Literal["SUBQUERY"] = "SUBQUERY"
    query: QueryNode


class FrameBoundType(str, Enum):
    UNBOUNDED_PRECEDING = "UNBOUNDED_PRECEDING"
    N_PRECEDING = "N_PRECEDING"
    CURRENT_ROW = "CURRENT_ROW"
    N_FOLLOWING = "N_FOLLOWING"
    UNBOUNDED_FOLLOWING = "UNBOUNDED_FOLLOWING"


class FrameBound(QueryIRV2Base):
    """Typed window frame bound."""
    bound_type: FrameBoundType
    offset: int | None = None

    @model_validator(mode="after")
    def _validate_offset(self) -> "FrameBound":
        needs_offset = self.bound_type in (FrameBoundType.N_PRECEDING, FrameBoundType.N_FOLLOWING)
        if needs_offset and (self.offset is None or self.offset < 0):
            raise ValueError(f"{self.bound_type.value} requires a non-negative offset")
        if not needs_offset and self.offset is not None:
            raise ValueError(f"{self.bound_type.value} must not have an offset")
        return self


class WindowSpecification(QueryIRV2Base):
    partition_by: list[Expression] = Field(default_factory=list)
    order_by: list[OrderByItem] = Field(default_factory=list)
    frame_type: Literal["ROWS", "RANGE"] | None = None
    frame_start: FrameBound | None = None
    frame_end: FrameBound | None = None
    # Legacy dict kept for backward compatibility with existing serializations
    frame: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_frame(self) -> "WindowSpecification":
        if self.frame_start is not None and self.frame_end is not None:
            start_order = list(FrameBoundType).index(self.frame_start.bound_type)
            end_order = list(FrameBoundType).index(self.frame_end.bound_type)
            if start_order > end_order:
                raise ValueError(
                    f"Frame start {self.frame_start.bound_type.value} "
                    f"must not come after end {self.frame_end.bound_type.value}"
                )
        return self


class WindowExpression(QueryIRV2Base):
    expression_type: Literal["WINDOW_EXPRESSION"] = "WINDOW_EXPRESSION"
    expression: Expression
    window: WindowSpecification = Field(default_factory=WindowSpecification)


class ComparisonPredicate(QueryIRV2Base):
    predicate_type: Literal["COMPARISON_PREDICATE"] = "COMPARISON_PREDICATE"
    left: Expression
    operator: Literal["=", "<>", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE"]
    right: Expression
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class InLiteralPredicate(QueryIRV2Base):
    predicate_type: Literal["IN_LITERAL_PREDICATE"] = "IN_LITERAL_PREDICATE"
    expression: Expression
    values: list[LiteralExpression]
    negated: bool = False
    legacy_v1: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_values(self) -> "InLiteralPredicate":
        if not self.values:
            raise ValueError("IN literal predicate requires at least one literal")
        return self


class BetweenPredicate(QueryIRV2Base):
    predicate_type: Literal["BETWEEN_PREDICATE"] = "BETWEEN_PREDICATE"
    expression: Expression
    lower: Expression
    upper: Expression
    negated: bool = False
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class NullPredicate(QueryIRV2Base):
    predicate_type: Literal["NULL_PREDICATE"] = "NULL_PREDICATE"
    expression: Expression
    negated: bool = False
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class BooleanPredicate(QueryIRV2Base):
    predicate_type: Literal["BOOLEAN_PREDICATE"] = "BOOLEAN_PREDICATE"
    operator: Literal["AND", "OR"]
    operands: list[Predicate]
    legacy_v1: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_predicate_count(self) -> "BooleanPredicate":
        if len(self.operands) < 2:
            raise ValueError(f"{self.operator} requires at least 2 predicates")
        return self


class NotPredicate(QueryIRV2Base):
    predicate_type: Literal["NOT_PREDICATE"] = "NOT_PREDICATE"
    operand: Predicate
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class InSubqueryPredicate(QueryIRV2Base):
    """IN / NOT IN with a subquery (distinct from InLiteralPredicate)."""
    predicate_type: Literal["IN_SUBQUERY_PREDICATE"] = "IN_SUBQUERY_PREDICATE"
    expression: Expression
    query: QueryNode
    negated: bool = False
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class ExistsPredicate(QueryIRV2Base):
    """EXISTS / NOT EXISTS predicate."""
    predicate_type: Literal["EXISTS_PREDICATE"] = "EXISTS_PREDICATE"
    query: QueryNode
    negated: bool = False
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class DateFilterNode(QueryIRV2Base):
    date_expression: Expression
    filter_type: Literal["relative_range", "absolute_range", "grain"]
    start_date: str | None = None
    end_date: str | None = None
    date_grain: str | None = None
    raw_text: str | None = None
    confidence: float = 1.0
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class SelectItem(QueryIRV2Base):
    expression: Expression
    alias: str | None = None
    name: str | None = None
    role: Literal["metric", "dimension", "date", "count", "record", "unknown"] = "unknown"
    source_slot: str | None = None
    confidence: float = 1.0
    legacy_v1: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FromItem(QueryIRV2Base):
    from_type: Literal["TABLE", "SUBQUERY"] = "TABLE"
    table: str | None = None
    alias: str | None = None
    query: QueryNode | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "FromItem":
        if self.from_type == "TABLE" and not self.table:
            raise ValueError("TABLE from_item requires table")
        if self.from_type == "SUBQUERY" and self.query is None:
            raise ValueError("SUBQUERY from_item requires query")
        return self


class JoinNode(QueryIRV2Base):
    join_type: Literal["INNER", "LEFT", "RIGHT", "FULL", "CROSS"] = "INNER"
    right: FromItem
    on: Predicate | None = None
    condition: str | None = None
    path_order: int = 0
    confidence: float = 1.0
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class OrderByItem(QueryIRV2Base):
    expression: Expression
    alias: str | None = None
    direction: Literal["ASC", "DESC"] = "ASC"
    source: Literal["metric", "dimension", "date", "count", "explicit", "default", "unknown"] = "unknown"
    nulls: Literal["FIRST", "LAST"] | None = None
    legacy_v1: dict[str, Any] = Field(default_factory=dict)


class SetOperationNode(QueryIRV2Base):
    operation: Literal["UNION", "UNION_ALL", "INTERSECT", "EXCEPT"]
    query: QueryNode


class CTEDefinition(QueryIRV2Base):
    """Common Table Expression definition (non-recursive)."""
    name: str
    columns: list[str] = Field(default_factory=list)
    query: QueryNode

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("CTE name must not be blank")
        return value


class QueryNode(QueryIRV2Base):
    query_ir_version: Literal["2.0"] = QUERY_IR_V2_VERSION
    query_type: Literal["SELECT"] = "SELECT"
    query_ir_id: str = ""
    question: str = ""
    normalized_question: str = ""
    intent: str = "show_records"
    template_id: str | None = None
    dialect: str = "sqlite"
    from_item: FromItem | None = None
    required_tables: list[str] = Field(default_factory=list)
    select_items: list[SelectItem] = Field(default_factory=list)
    joins: list[JoinNode] = Field(default_factory=list)
    where: Predicate | None = None
    having: Predicate | None = None
    predicates: list[Predicate] = Field(default_factory=list)
    date_filters: list[DateFilterNode] = Field(default_factory=list)
    group_by: list[Expression] = Field(default_factory=list)
    order_by: list[OrderByItem] = Field(default_factory=list)
    limit: int | None = 100
    offset: int | None = None
    ctes: list[CTEDefinition] = Field(default_factory=list)
    set_operations: list[SetOperationNode] = Field(default_factory=list)
    select_mode: Literal["records", "aggregate", "trend", "count"] = "records"
    warnings: list[str] = Field(default_factory=list)
    capability_metadata: CapabilityMetadata = Field(default_factory=CapabilityMetadata)
    confidence_metadata: ConfidenceMetadata = Field(default_factory=ConfidenceMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("limit", "offset")
    @classmethod
    def _non_negative_int(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("limit and offset must be non-negative")
        return value


Expression: TypeAlias = Annotated[
    ColumnExpression
    | LiteralExpression
    | FunctionExpression
    | AggregationExpression
    | BinaryOperationExpression
    | UnaryOperationExpression
    | BooleanOperationExpression
    | CaseExpression
    | SubqueryExpression
    | WindowExpression,
    Field(discriminator="expression_type"),
]

Predicate: TypeAlias = Annotated[
    ComparisonPredicate
    | InLiteralPredicate
    | BetweenPredicate
    | NullPredicate
    | BooleanPredicate
    | NotPredicate
    | InSubqueryPredicate
    | ExistsPredicate,
    Field(discriminator="predicate_type"),
]

InPredicate = InLiteralPredicate


_NAMESPACE = {
    "Expression": Expression,
    "Predicate": Predicate,
    "QueryNode": QueryNode,
    "OrderByItem": OrderByItem,
}

for _model in [
    FunctionExpression,
    AggregationExpression,
    BinaryOperationExpression,
    UnaryOperationExpression,
    BooleanOperationExpression,
    CaseWhen,
    CaseExpression,
    SubqueryExpression,
    FrameBound,
    WindowSpecification,
    WindowExpression,
    ComparisonPredicate,
    InLiteralPredicate,
    BetweenPredicate,
    NullPredicate,
    BooleanPredicate,
    NotPredicate,
    InSubqueryPredicate,
    ExistsPredicate,
    DateFilterNode,
    SelectItem,
    FromItem,
    JoinNode,
    OrderByItem,
    SetOperationNode,
    CTEDefinition,
    QueryNode,
]:
    _model.model_rebuild(_types_namespace=_NAMESPACE)


__all__ = [
    "QUERY_IR_V2_VERSION",
    "AggregationExpression",
    "BetweenPredicate",
    "BinaryOperationExpression",
    "BooleanOperationExpression",
    "BooleanPredicate",
    "CTEDefinition",
    "CapabilityMetadata",
    "CaseExpression",
    "CaseWhen",
    "ColumnExpression",
    "ComparisonPredicate",
    "ConfidenceMetadata",
    "DateFilterNode",
    "ExistsPredicate",
    "Expression",
    "FrameBound",
    "FrameBoundType",
    "FromItem",
    "FunctionExpression",
    "InLiteralPredicate",
    "InPredicate",
    "InSubqueryPredicate",
    "JoinNode",
    "LiteralExpression",
    "LiteralValue",
    "LiteralValueType",
    "NullPredicate",
    "NotPredicate",
    "OrderByItem",
    "Predicate",
    "QueryNode",
    "SelectItem",
    "SetOperationNode",
    "SubqueryExpression",
    "UnaryOperationExpression",
    "WindowExpression",
    "WindowSpecification",
]
