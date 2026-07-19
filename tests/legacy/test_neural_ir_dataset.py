"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json

from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch
from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.tokenizer import tokenize
from neural_ir.vocab import Vocabulary


def test_ir_training_dataset_loads_small_jsonl(tmp_path) -> None:
    path = tmp_path / "ir.jsonl"
    row = _row()
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    vocab = Vocabulary()
    vocab.build([tokenize(row["question"]), tokenize(row["serialized_schema"])])
    dataset = IRTrainingDataset(str(path), vocab, IRLabelEncoder(), max_question_len=12, max_schema_len=24)

    item = dataset[0]
    batch = collate_ir_batch([item])

    assert len(dataset) == 1
    assert batch["question_ids"].shape == (1, 12)
    assert batch["relation_type_ids"].shape == (1, 12, 24)
    assert batch["schema_relation_type_ids"].shape == (1, 24, 24)
    assert batch["labels"]["intent_label"].shape == (1,)
    assert item["schema_items"]["tables"] == ["orders"]


def test_ir_training_dataset_pairwise_relations_tolerate_missing_fk(tmp_path) -> None:
    path = tmp_path / "ir.jsonl"
    row = _row()
    row["serialized_schema"] = "tables: orders(order_id, customer_id, amount); customers(id, name)"
    row["query_ir"]["metadata"]["validation_context"]["schema_context"] = {
        "tables": {
            "orders": {
                "columns": {
                    "order_id": {"primary_key": True},
                    "customer_id": {},
                    "amount": {"type": "numeric"},
                }
            },
            "customers": {"columns": {"id": {"primary_key": True}, "name": {"type": "text"}}},
        }
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    vocab = Vocabulary()
    vocab.build([tokenize(row["question"]), tokenize(row["serialized_schema"])])
    dataset = IRTrainingDataset(str(path), vocab, IRLabelEncoder(), max_question_len=12, max_schema_len=32)

    item = dataset[0]

    assert len(item["schema_relation_type_ids"]) == 32
    assert len(item["schema_relation_type_ids"][0]) == 32


def _row() -> dict:
    return {
        "example_id": "x1",
        "question": "How many orders?",
        "serialized_schema": "tables: orders(order_id, amount)",
        "query_ir": {
            "intent": "count_records",
            "template_id": "count_records",
            "base_table": "orders",
            "required_tables": ["orders"],
            "metrics": [{"aggregation": "COUNT", "table": "orders", "column": "*", "expression": "*"}],
            "dimensions": [],
            "filters": [],
            "date_filters": [],
            "order_by": [],
            "limit": 100,
            "metadata": {"validation_context": {"schema_context": {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}}},
        },
    }
