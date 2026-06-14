from __future__ import annotations

from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator
from neural_ir.option_a_to_ir import OptionAToIRConverter
from validation.sql_validator import SQLValidator


def test_option_a_prediction_converts_to_valid_query_ir() -> None:
    schema = _retail_schema()
    decoded = {
        "intent": "top_n_metric_by_dimension",
        "template_id": "top_n_metric_by_dimension",
        "base_table": "orders",
        "metric_aggregation": "SUM",
        "metric_column": {"table": "orders", "column": "amount", "type": "numeric"},
        "metric_expression_type": "column",
        "dimension_column": {"table": "customers", "column": "customer_name", "type": "text"},
        "date_column": None,
        "date_grain": "none",
        "date_filter_type": "none",
        "filter_column": None,
        "filter_operator": "none",
        "order_direction": "DESC",
        "limit": 5,
    }

    query_ir = OptionAToIRConverter().convert("Top 5 customers by sales", schema, decoded)
    ir_validation = IRValidator().validate(query_ir, schema=schema)
    sql = IRToSQLRenderer().render(query_ir)
    sql_validation = SQLValidator().validate(sql, schema=schema)

    assert ir_validation.is_valid
    assert sql_validation["is_valid"]
    assert "SUM(orders.amount)" in sql


def _retail_schema() -> dict:
    return {
        "dialect": "sqlite",
        "tables": {
            "orders": {
                "columns": {
                    "order_id": {"type": "INTEGER"},
                    "customer_id": {"type": "INTEGER"},
                    "order_date": {"type": "DATE"},
                    "amount": {"type": "FLOAT"},
                    "status": {"type": "TEXT"},
                }
            },
            "customers": {
                "columns": {
                    "customer_id": {"type": "INTEGER"},
                    "customer_name": {"type": "TEXT"},
                    "region": {"type": "TEXT"},
                }
            },
        },
        "foreign_keys": [
            {
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers",
                "to_column": "customer_id",
            }
        ],
    }
