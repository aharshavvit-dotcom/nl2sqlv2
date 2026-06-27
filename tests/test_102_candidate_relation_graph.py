from __future__ import annotations

import json
import torch
import pytest

from neural_ir.ir_dataset import (
    IRTrainingDataset,
    collate_ir_batch,
    build_candidate_metadata,
    build_candidate_pairwise_relation_matrix,
    RELATION_TYPE_MAP,
)
from neural_ir.attention_model import SchemaAwareOptionAIRModel
from neural_ir.trainer import MODEL_INPUT_KEYS
from neural_ir.vocab import Vocabulary
from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_optimization.training_diagnostics import TrainingDiagnostics


def test_unified_candidate_indices_no_collision():
    # Table candidates and Column candidates should not collide on candidate_index
    candidates = {
        "tables": [{"table": "users", "index": 0}, {"table": "orders", "index": 1}],
        "columns": [
            {"table": "users", "column": "id", "index": 0},
            {"table": "orders", "column": "id", "index": 1},
        ],
    }
    schema_items = {
        "columns": [
            {"table": "users", "column": "id", "type": "integer", "primary_key": True},
            {"table": "orders", "column": "id", "type": "integer", "primary_key": True},
        ]
    }
    max_tables = 10
    
    metadata = build_candidate_metadata(candidates, schema_items, max_tables=max_tables)
    
    # 2 tables + 2 columns
    assert len(metadata) == 4
    
    indices = [item["candidate_index"] for item in metadata]
    assert len(indices) == len(set(indices)), f"Indices collided: {indices}"
    # Table indices: 0, 1
    # Column indices: max_tables + 0, max_tables + 1
    assert indices == [0, 1, 10, 11]
    
    # Verify candidate_ids are distinct for duplicate column names ("id") across different tables
    cand_ids = [item["candidate_id"] for item in metadata]
    assert "column:users.id" in cand_ids
    assert "column:orders.id" in cand_ids
    assert len(set(cand_ids)) == 4


def test_table_has_column_relation():
    # Test that build_candidate_pairwise_relation_matrix creates table_has_column relation
    metadata = [
        {
            "candidate_id": "table:users",
            "candidate_type": "table",
            "candidate_index": 0,
            "table_name": "users",
        },
        {
            "candidate_id": "column:users.id",
            "candidate_type": "column",
            "candidate_index": 10,
            "table_name": "users",
            "column_name": "id",
        },
    ]
    matrix = build_candidate_pairwise_relation_matrix(metadata, max_candidates=20)
    # Left: table:users (0), Right: column:users.id (10)
    assert matrix[0][10] == RELATION_TYPE_MAP["table_has_column"]
    # Left: column:users.id (10), Right: table:users (0)
    assert matrix[10][0] == RELATION_TYPE_MAP["column_belongs_to_table"]


def test_fk_relations():
    # Test pk_to_fk and fk_to_pk relations
    metadata = [
        {
            "candidate_id": "column:users.id",
            "candidate_type": "column",
            "candidate_index": 10,
            "table_name": "users",
            "column_name": "id",
            "is_primary_key": True,
        },
        {
            "candidate_id": "column:orders.user_id",
            "candidate_type": "column",
            "candidate_index": 11,
            "table_name": "orders",
            "column_name": "user_id",
            "is_foreign_key": True,
            "foreign_key_target": {"table": "users", "column": "id"},
        },
    ]
    matrix = build_candidate_pairwise_relation_matrix(metadata, max_candidates=20)
    # left: column:orders.user_id (11) -> right: column:users.id (10) should be fk_to_pk
    assert matrix[11][10] == RELATION_TYPE_MAP["fk_to_pk"]
    # left: column:users.id (10) -> right: column:orders.user_id (11) should be pk_to_fk
    assert matrix[10][11] == RELATION_TYPE_MAP["pk_to_fk"]


def test_model_input_keys_includes_candidate_relation():
    assert "candidate_relation_type_ids" in MODEL_INPUT_KEYS


def test_candidate_relation_type_ids_flow_dataset_to_collate(tmp_path):
    path = tmp_path / "ir.jsonl"
    row = {
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
            "metadata": {
                "validation_context": {
                    "schema_context": {
                        "tables": {
                            "orders": {
                                "columns": {
                                    "order_id": {"primary_key": True},
                                    "amount": {"type": "numeric"},
                                }
                            }
                        }
                    }
                }
            },
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    
    vocab = Vocabulary()
    vocab.build([["How", "many", "orders?"], ["tables:", "orders", "(", "order_id", ",", "amount", ")"]])
    
    max_tables = 5
    max_columns = 10
    max_candidates = max_tables + max_columns
    
    dataset = IRTrainingDataset(
        str(path),
        vocab,
        IRLabelEncoder(),
        max_question_len=8,
        max_schema_len=16,
        max_tables=max_tables,
        max_columns=max_columns,
    )
    
    item = dataset[0]
    assert "candidate_relation_type_ids" in item
    assert len(item["candidate_relation_type_ids"]) == max_candidates
    assert len(item["candidate_relation_type_ids"][0]) == max_candidates
    
    batch = collate_ir_batch([item])
    assert "candidate_relation_type_ids" in batch
    # Shape should be [batch_size, max_candidates, max_candidates]
    assert batch["candidate_relation_type_ids"].shape == (1, max_candidates, max_candidates)


def test_relation_bias_changes_output():
    # Prove that enabling vs disabling candidate relation bias changes output candidate logits/scores when bias is non-zero
    config = {
        "hidden_dim": 16,
        "max_tables": 4,
        "max_columns": 8,
        "relation_aware_attention": {
            "enabled": True,
            "relation_bias_mode": "candidate_pairwise_relation_bias",
            "bias_init": 0.0,
        },
    }
    
    vocab_size = 100
    label_sizes = {
        "intent": 5,
        "metric_aggregation": 3,
        "metric_expression_type": 2,
        "date_grain": 2,
        "date_filter_type": 2,
        "filter_operator": 4,
        "limit_bucket": 3,
        "order_direction": 2,
    }
    
    model = SchemaAwareOptionAIRModel(config, vocab_size, label_sizes)
    
    # Enable it and set a non-zero bias
    model.relation_aware_enabled = True
    model.relation_bias_mode = "candidate_pairwise_relation_bias"
    # Manually fill the relation bias with non-zero values to ensure change in logits is visible
    model.relation_bias.bias.data.fill_(5.5)
    
    batch_size = 1
    question_ids = torch.randint(1, vocab_size, (batch_size, 8))
    schema_ids = torch.randint(1, vocab_size, (batch_size, 16))
    table_candidate_token_ids = torch.randint(1, vocab_size, (batch_size, 4, 3))
    column_candidate_token_ids = torch.randint(1, vocab_size, (batch_size, 8, 3))
    candidate_relation_type_ids = torch.randint(0, len(RELATION_TYPE_MAP), (batch_size, 12, 12))
    
    # Run with non-zero bias
    out_biased = model(
        question_ids=question_ids,
        schema_ids=schema_ids,
        table_candidate_token_ids=table_candidate_token_ids,
        column_candidate_token_ids=column_candidate_token_ids,
        candidate_relation_type_ids=candidate_relation_type_ids,
    )
    
    # Now run without bias (bias set to zero)
    model.relation_bias.bias.data.fill_(0.0)
    out_unbiased = model(
        question_ids=question_ids,
        schema_ids=schema_ids,
        table_candidate_token_ids=table_candidate_token_ids,
        column_candidate_token_ids=column_candidate_token_ids,
        candidate_relation_type_ids=candidate_relation_type_ids,
    )
    
    # Confirm that logits/scores are different when bias is non-zero
    biased_table_logits = out_biased["base_table_logits"]
    unbiased_table_logits = out_unbiased["base_table_logits"]
    
    # They should not be equal
    assert not torch.allclose(biased_table_logits, unbiased_table_logits)


def test_candidate_relation_diagnostics_report_candidate_graph():
    diag = TrainingDiagnostics()
    diag.set_config({
        "model": {
            "relation_aware_attention": {
                "enabled": True,
                "relation_bias_mode": "candidate_pairwise_relation_bias",
                "relation_types": ["same_table", "table_has_column"],
            }
        }
    })
    
    summary = diag.config_summary
    assert "relation_aware_attention" in summary
    rat = summary["relation_aware_attention"]
    assert rat["enabled"] is True
    assert rat["candidate_relation_type_ids_available"] is True
    assert rat["candidate_pairwise_relation_bias_active"] is True
