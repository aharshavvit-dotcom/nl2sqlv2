from __future__ import annotations

import json

from neural_ir.ir_dataset import IRTrainingDataset
from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.tokenizer import tokenize
from neural_ir.vocab import Vocabulary


def test_option_a_dataset_candidate_masks(tmp_path) -> None:
    row = _row()
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    vocab = Vocabulary()
    vocab.build([tokenize(row["question"]), tokenize(row["serialized_schema"])])

    item = IRTrainingDataset(str(path), vocab, IRLabelEncoder(), max_question_len=16, max_schema_len=64, max_columns=16)[0]
    columns = item["schema_items"]["columns"]
    index = {f"{col['table']}.{col['column']}": col["index"] for col in columns}

    assert item["metric_column_mask"][index["orders.amount"]] == 1
    assert item["metric_column_mask"][index["customers.customer_name"]] == 0
    assert item["dimension_column_mask"][index["customers.customer_name"]] == 1
    assert item["dimension_column_mask"][index["orders.status"]] == 1
    assert item["date_column_mask"][index["orders.order_date"]] == 1
    assert item["date_column_mask"][index["orders.amount"]] == 0
    assert item["filter_column_mask"][index["orders.status"]] == 1
    assert item["filter_column_mask"][index["customers.region"]] == 1
    assert item["filter_column_mask"][index["orders.order_date"]] == 1


def _row() -> dict:
    return {
        "example_id": "wikisql:x1",
        "question": "Top customers by sales",
        "serialized_schema": "tables: orders(order_id, amount, order_date, status); customers(customer_id, customer_name, region)",
        "query_ir": {
            "intent": "top_n_metric_by_dimension",
            "template_id": "top_n_metric_by_dimension",
            "base_table": "orders",
            "required_tables": ["orders", "customers"],
            "metrics": [{"aggregation": "SUM", "table": "orders", "column": "amount", "expression": "orders.amount"}],
            "dimensions": [{"table": "customers", "column": "customer_name", "expression": "customers.customer_name"}],
            "filters": [],
            "date_filters": [],
            "order_by": [{"direction": "DESC"}],
            "limit": 5,
        },
    }

