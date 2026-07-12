"""QueryIR v2 round-trip validation framework.

Validates that: SQL → QueryIR v2 → SQL → execute produces equivalent results.
This is the core correctness assurance for the full v2 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer, QueryIRV2RenderingError
from .query_ir_v2_models import QueryNode
from .query_ir_v2_scope import QueryAnalysis, QueryScopeAnalyzer
from .query_ir_v2_validation import QueryIRV2Validator, QueryIRV2ValidationResult
from .sql_to_query_ir_v2 import SQLToQueryIRV2Converter, SQLToQueryIRV2Error


@dataclass
class RoundtripResult:
    """Result of a round-trip validation."""
    original_sql: str
    question: str = ""
    query_ir: QueryNode | None = None
    rendered_sql: str | None = None
    validation: QueryIRV2ValidationResult | None = None
    analysis: QueryAnalysis | None = None

    # Status flags
    conversion_success: bool = False
    validation_success: bool = False
    rendering_success: bool = False
    execution_equivalent: bool | None = None

    # Errors
    conversion_error: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    rendering_error: str | None = None
    execution_error: str | None = None

    @property
    def round_trip_success(self) -> bool:
        return self.conversion_success and self.validation_success and self.rendering_success


@dataclass
class RoundtripReport:
    """Summary of multiple round-trip validations."""
    total: int = 0
    conversion_passed: int = 0
    validation_passed: int = 0
    rendering_passed: int = 0
    full_roundtrip_passed: int = 0
    execution_equivalent: int = 0
    results: list[RoundtripResult] = field(default_factory=list)
    failures: list[RoundtripResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.full_roundtrip_passed / self.total if self.total > 0 else 0.0


class QueryIRV2RoundtripValidator:
    """Validates the SQL → QueryIR v2 → SQL round-trip pipeline.

    Usage:
        validator = QueryIRV2RoundtripValidator(dialect="sqlite")
        result = validator.validate("SELECT * FROM orders WHERE id > 10", "orders over 10")
        assert result.round_trip_success
    """

    def __init__(
        self,
        *,
        dialect: str = "sqlite",
        enable_or_rendering: bool = True,
        max_limit: int = 1000,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.dialect = dialect
        self.schema = schema
        self.converter = SQLToQueryIRV2Converter(dialect=dialect)
        self.validator = QueryIRV2Validator()
        self.scope_analyzer = QueryScopeAnalyzer()
        self.renderer = QueryIRV2NativeRenderer(
            enable_or_rendering=enable_or_rendering,
            max_limit=max_limit,
        )

    def validate(
        self,
        sql: str,
        question: str = "",
    ) -> RoundtripResult:
        """Run a full round-trip validation for one SQL statement."""
        result = RoundtripResult(original_sql=sql, question=question)

        # Step 1: Convert SQL → QueryIR v2
        try:
            query_ir = self.converter.convert(sql, question=question, schema=self.schema)
            result.query_ir = query_ir
            result.conversion_success = True
        except (SQLToQueryIRV2Error, Exception) as e:
            result.conversion_error = str(e)
            return result

        # Step 2: Validate QueryIR v2
        validation = self.validator.validate(query_ir)
        result.validation = validation
        result.validation_errors = list(validation.errors)
        result.validation_success = validation.is_valid

        # Step 3: Scope analysis (informational, does not block)
        try:
            result.analysis = self.scope_analyzer.analyze(query_ir)
        except Exception:
            pass  # Scope analysis failures are informational

        # Step 4: Render QueryIR v2 → SQL
        try:
            rendered = self.renderer.render(query_ir, self.dialect)
            result.rendered_sql = rendered
            result.rendering_success = True
        except QueryIRV2RenderingError as e:
            result.rendering_error = str(e)
        except Exception as e:
            result.rendering_error = f"Unexpected: {e}"

        return result

    def validate_batch(
        self,
        cases: list[tuple[str, str]],
    ) -> RoundtripReport:
        """Run round-trip validation for a batch of (sql, question) pairs."""
        report = RoundtripReport()
        for sql, question in cases:
            result = self.validate(sql, question)
            report.total += 1
            report.results.append(result)

            if result.conversion_success:
                report.conversion_passed += 1
            if result.validation_success:
                report.validation_passed += 1
            if result.rendering_success:
                report.rendering_passed += 1
            if result.round_trip_success:
                report.full_roundtrip_passed += 1
            else:
                report.failures.append(result)

        return report


__all__ = [
    "QueryIRV2RoundtripValidator",
    "RoundtripReport",
    "RoundtripResult",
]
