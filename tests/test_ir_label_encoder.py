from __future__ import annotations

from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.schema_linearizer import extract_schema_items


def test_ir_label_encoder_encodes_and_decodes_query_ir() -> None:
    schema = {
        "tables": {
            "orders": {"columns": {"order_id": {"type": "INTEGER"}, "amount": {"type": "FLOAT"}}},
            "customers": {"columns": {"customer_name": {"type": "TEXT"}}},
        }
    }
    schema_items = extract_schema_items(schema)
    query_ir = {
        "intent": "top_n_metric_by_dimension",
        "template_id": "top_n_metric_by_dimension",
        "base_table": "orders",
        "metrics": [{"aggregation": "SUM", "table": "orders", "column": "amount", "expression": "orders.amount"}],
        "dimensions": [{"table": "customers", "column": "customer_name"}],
        "date_filters": [],
        "filters": [],
        "order_by": [{"direction": "DESC"}],
        "limit": 5,
    }

    encoder = IRLabelEncoder()
    labels = encoder.encode(query_ir, schema_items)
    decoded = encoder.decode(labels, schema_items)

    assert labels["base_table_index"] == 0
    assert decoded["intent"] == "top_n_metric_by_dimension"
    assert decoded["metric_column"]["column"] == "amount"
    assert decoded["dimension_column"]["column"] == "customer_name"
    assert decoded["limit"] == 5
