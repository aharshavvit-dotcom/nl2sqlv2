from __future__ import annotations

from ir.ir_validator import IRValidator
from ir.query_ir_models import IRMetric, QueryIR


def test_ir_validator_accepts_valid_metric_summary() -> None:
    query_ir = QueryIR(
        query_ir_id="qir-test",
        question="Show sales",
        normalized_question="show sales",
        intent="metric_summary",
        template_id="metric_summary",
        base_table="orders",
        required_tables=["orders"],
        metrics=[
            IRMetric(
                name="sales",
                aggregation="SUM",
                table="orders",
                column="amount",
                expression="orders.amount",
                alias="revenue",
            )
        ],
        limit=100,
        select_mode="aggregate",
    )

    result = IRValidator().validate(query_ir, schema={"tables": {"orders": {"columns": {"amount": {}}}}})
    assert result.is_valid


def test_ir_validator_rejects_missing_metric() -> None:
    query_ir = QueryIR(
        query_ir_id="qir-test",
        question="Show sales",
        normalized_question="show sales",
        intent="metric_summary",
        template_id="metric_summary",
        base_table="orders",
        required_tables=["orders"],
        limit=100,
        select_mode="aggregate",
    )

    result = IRValidator().validate(query_ir)
    assert not result.is_valid
    assert any("Metric intent" in error for error in result.errors)
