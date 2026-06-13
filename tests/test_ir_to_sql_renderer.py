from __future__ import annotations

from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.query_ir_models import IRDimension, IRMetric, IROrderBy, QueryIR


def test_ir_to_sql_renderer_renders_metric_by_dimension() -> None:
    query_ir = QueryIR(
        query_ir_id="qir-test",
        question="Sales by region",
        normalized_question="sales by region",
        intent="metric_by_dimension",
        template_id="metric_by_dimension",
        base_table="orders",
        required_tables=["orders", "customers"],
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
        dimensions=[
            IRDimension(
                name="region",
                table="customers",
                column="region",
                expression="customers.region",
                alias="region",
            )
        ],
        group_by=["customers.region"],
        order_by=[IROrderBy(expression="revenue", alias="revenue", direction="DESC", source="metric")],
        limit=100,
        select_mode="aggregate",
    )

    sql = IRToSQLRenderer().render(query_ir)
    assert "customers.region AS region" in sql
    assert "SUM(orders.amount) AS revenue" in sql
    assert "GROUP BY customers.region" in sql
    assert "LIMIT 100" in sql
