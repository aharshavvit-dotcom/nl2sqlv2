"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from ir.option_c_to_ir import OptionCToIRConverter
from inference.prediction_models import JoinPlan, SchemaMapping


def test_option_c_to_ir_converts_top_metric_by_dimension() -> None:
    mapping = SchemaMapping(
        metric_name="sales",
        metric_table="orders",
        metric_column="amount",
        metric_aggregation="SUM",
        dimension_name="customer",
        dimension_table="customers",
        dimension_column="customer_name",
        entity_table="orders",
        date_table="orders",
        date_column="order_date",
    )
    join_plan = JoinPlan(
        base_table="orders",
        required_tables=["orders", "customers"],
        join_steps=[
            {
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers",
                "to_column": "customer_id",
                "current": "orders",
                "neighbor": "customers",
                "condition": "orders.customer_id = customers.customer_id",
            }
        ],
    )

    query_ir = OptionCToIRConverter().convert(
        question="Top 5 customers by sales",
        normalized_question="top 5 customers by sales",
        intent="top_n_metric_by_dimension",
        template_id="top_n_metric_by_dimension",
        slots={"limit": {"value": 5}, "metric": {"value": "sales", "confidence": 0.9}, "dimension": {"value": "customer", "confidence": 0.9}},
        schema_mapping=mapping,
        join_plan=join_plan,
    )

    assert query_ir.metrics[0].expression == "orders.amount"
    assert query_ir.dimensions[0].expression == "customers.customer_name"
    assert query_ir.joins[0].right_table == "customers"
    assert query_ir.order_by[0].direction == "DESC"
    assert query_ir.limit == 5
