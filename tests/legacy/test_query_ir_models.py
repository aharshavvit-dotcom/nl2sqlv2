from __future__ import annotations

from ir.query_ir_models import IRMetric, QueryIR


def test_query_ir_models_serialize_metric() -> None:
    metric = IRMetric(
        name="revenue",
        aggregation="SUM",
        table="orders",
        column="amount",
        expression="orders.amount",
        alias="revenue",
    )
    query_ir = QueryIR(
        query_ir_id="qir-test",
        question="Top customers by sales",
        normalized_question="top customers by sales",
        intent="metric_summary",
        template_id="metric_summary",
        base_table="orders",
        required_tables=["orders"],
        metrics=[metric],
        limit=100,
        select_mode="aggregate",
    )

    payload = query_ir.model_dump()
    assert payload["metrics"][0]["expression"] == "orders.amount"
    assert payload["select_mode"] == "aggregate"
