"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import torch

from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.model import OptionAIRModel
from neural_ir.model_registry import save_model_bundle
from neural_ir.predictor import OptionAIRPredictor
from neural_ir.schema_linearizer import extract_schema_items
from neural_ir.tokenizer import tokenize
from neural_ir.vocab import Vocabulary


def test_option_a_predictor_loads_tiny_bundle_and_predicts(tmp_path) -> None:
    schema = _retail_schema()
    schema_items = extract_schema_items(schema)
    encoder = IRLabelEncoder()
    vocab = Vocabulary()
    vocab.build([tokenize("Top 5 customers by sales"), tokenize("tables orders amount customers customer_name")])
    config = {"embedding_dim": 8, "hidden_dim": 8, "max_tables": 8, "max_columns": 16, "max_question_len": 16, "max_schema_len": 32}
    model = OptionAIRModel(config, vocab_size=len(vocab), label_sizes=encoder.label_sizes)
    _bias_model_for_top_customers(model, encoder, schema_items)
    save_model_bundle(model, vocab, encoder, config, tmp_path)

    result = OptionAIRPredictor(str(tmp_path)).predict("Top 5 customers by sales", schema)

    assert result["query_ir"]["template_id"] == "top_n_metric_by_dimension"
    assert result["ir_validation"]["is_valid"]
    assert result["sql_validation"]["is_valid"]


def _bias_model_for_top_customers(model, encoder: IRLabelEncoder, schema_items: dict) -> None:
    for param in model.parameters():
        param.data.zero_()
    column_index = {
        (item["table"], item["column"]): idx
        for idx, item in enumerate(schema_items["columns"])
    }
    model.intent_head.bias.data[encoder.label_maps["intent"]["top_n_metric_by_dimension"]] = 5
    model.base_table_head.bias.data[schema_items["tables"].index("orders")] = 5
    model.metric_aggregation_head.bias.data[encoder.label_maps["metric_aggregation"]["SUM"]] = 5
    model.metric_column_head.bias.data[column_index[("orders", "amount")]] = 5
    model.metric_expression_type_head.bias.data[encoder.label_maps["metric_expression_type"]["column"]] = 5
    model.dimension_column_head.bias.data[column_index[("customers", "customer_name")]] = 5
    model.date_column_head.bias.data[0] = 5
    model.date_grain_head.bias.data[encoder.label_maps["date_grain"]["none"]] = 5
    model.date_filter_type_head.bias.data[encoder.label_maps["date_filter_type"]["none"]] = 5
    model.filter_column_head.bias.data[0] = 5
    model.filter_operator_head.bias.data[encoder.label_maps["filter_operator"]["none"]] = 5
    model.order_direction_head.bias.data[encoder.label_maps["order_direction"]["DESC"]] = 5
    model.limit_bucket_head.bias.data[encoder.label_maps["limit_bucket"]["top_5"]] = 5


def _retail_schema() -> dict:
    return {
        "dialect": "sqlite",
        "tables": {
            "orders": {
                "columns": {
                    "order_id": {"type": "INTEGER"},
                    "customer_id": {"type": "INTEGER"},
                    "amount": {"type": "FLOAT"},
                }
            },
            "customers": {
                "columns": {
                    "customer_id": {"type": "INTEGER"},
                    "customer_name": {"type": "TEXT"},
                }
            },
        },
        "foreign_keys": [{"from_table": "orders", "from_column": "customer_id", "to_table": "customers", "to_column": "customer_id"}],
    }
