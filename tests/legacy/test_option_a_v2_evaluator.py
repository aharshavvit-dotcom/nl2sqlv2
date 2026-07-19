"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json

from neural_ir.attention_model import SchemaAwareOptionAIRModel
from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.model_registry import save_model_bundle
from neural_ir.tokenizer import tokenize
from neural_ir.vocab import Vocabulary
from training_ir.evaluate_option_a_v2_model import evaluate_option_a_v2_model


def test_option_a_v2_evaluator_writes_report_structure(tmp_path) -> None:
    model_dir = tmp_path / "model"
    test_path = tmp_path / "test.jsonl"
    output_path = tmp_path / "report.json"
    row = _row()
    test_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    encoder = IRLabelEncoder()
    vocab = Vocabulary()
    vocab.build([tokenize(row["question"]), tokenize(row["serialized_schema"])])
    config = {"model_version": "option_a_v2", "embedding_dim": 8, "hidden_dim": 8, "candidate_hidden_dim": 6, "max_tables": 4, "max_columns": 8, "max_schema_len": 32}
    model = SchemaAwareOptionAIRModel(config, len(vocab), encoder.label_sizes)
    save_model_bundle(model, vocab, encoder, config, model_dir)

    report = evaluate_option_a_v2_model(model_dir, test_path, output_path)

    assert "summary" in report
    assert "by_intent" in report
    assert "repair_stats" in report
    assert output_path.exists()


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
            "metrics": [{"name": "record_count", "aggregation": "COUNT", "table": "orders", "column": "*", "expression": "*", "alias": "record_count"}],
            "dimensions": [],
            "filters": [],
            "date_filters": [],
            "order_by": [],
            "limit": 100,
            "metadata": {"validation_context": {"schema_context": {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}}},
        },
    }
