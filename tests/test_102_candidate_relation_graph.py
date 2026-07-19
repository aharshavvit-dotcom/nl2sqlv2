"""
Purpose: Protects ir unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

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


def _label_sizes():
    return {
        "intent": 5,
        "metric_aggregation": 3,
        "metric_expression_type": 2,
        "date_grain": 2,
        "date_filter_type": 2,
        "filter_operator": 4,
        "limit_bucket": 3,
        "order_direction": 2,
    }


def _relation_model(mode="candidate_pairwise_relation_bias", enabled=True):
    torch.manual_seed(42)
    model = SchemaAwareOptionAIRModel({
        "embedding_dim": 8,
        "hidden_dim": 4,
        "candidate_hidden_dim": 4,
        "dropout": 0.0,
        "max_tables": 2,
        "max_columns": 2,
        "relation_aware_attention": {
            "enabled": enabled,
            "relation_bias_mode": mode,
            "bias_init": 0.0,
        },
    }, 50, _label_sizes())
    model.eval()
    return model


def _forward_inputs():
    torch.manual_seed(42)
    return {
        "question_ids": torch.randint(1, 50, (1, 3)),
        "schema_ids": torch.randint(1, 50, (1, 4)),
        "question_mask": torch.ones((1, 3)),
        "schema_mask": torch.ones((1, 4)),
        "table_candidate_token_ids": torch.randint(1, 50, (1, 2, 2)),
        "column_candidate_token_ids": torch.randint(1, 50, (1, 2, 2)),
        "table_candidate_mask": torch.ones((1, 2)),
        "column_candidate_mask": torch.ones((1, 2)),
    }


def test_candidate_pairwise_attention_uses_candidate_mask():
    model = _relation_model()
    with torch.no_grad():
        model.relation_bias.bias.copy_(torch.arange(model.relation_bias.num_types).float())
        vectors = torch.randn(1, 4, 8)
        relation_ids = torch.randint(0, model.relation_bias.num_types, (1, 4, 4))
        masked = model._candidate_pairwise_relation_context(
            vectors, torch.tensor([[1, 1, 0, 0]]), relation_ids,
        )
        full = model._candidate_pairwise_relation_context(
            vectors, torch.ones((1, 4)), relation_ids,
        )
    assert not torch.allclose(masked[:, :2], full[:, :2])
    assert torch.count_nonzero(masked[:, 2:]) == 0


def test_padded_candidates_do_not_change_valid_candidate_outputs():
    model = _relation_model()
    with torch.no_grad():
        model.relation_bias.bias.copy_(torch.arange(model.relation_bias.num_types).float())
        vectors = torch.randn(1, 4, 8)
        changed_padding = vectors.clone()
        changed_padding[:, 2:] = torch.randn_like(changed_padding[:, 2:]) * 1000
        mask = torch.tensor([[1, 1, 0, 0]])
        relation_ids = torch.randint(0, model.relation_bias.num_types, (1, 4, 4))
        original = model._candidate_pairwise_relation_context(vectors, mask, relation_ids)
        changed = model._candidate_pairwise_relation_context(changed_padding, mask, relation_ids)
    assert torch.allclose(original[:, :2], changed[:, :2])


def test_candidate_mask_shape_mismatch_fails_clearly():
    model = _relation_model()
    with pytest.raises(ValueError, match="candidate_mask must have shape"):
        model._candidate_pairwise_relation_context(
            torch.randn(1, 4, 8),
            torch.ones((1, 3)),
            torch.zeros((1, 4, 4), dtype=torch.long),
        )


def test_candidate_relation_attention_accepts_full_valid_mask():
    model = _relation_model()
    result = model._candidate_pairwise_relation_context(
        torch.randn(1, 4, 8),
        torch.ones((1, 4)),
        torch.zeros((1, 4, 4), dtype=torch.long),
    )
    assert result.shape == (1, 4, 8)


def test_relation_bias_changes_output_deterministically():
    model = _relation_model()
    inputs = _forward_inputs()
    relation_ids = torch.tensor([[
        [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6],
    ]])
    with torch.no_grad():
        model.relation_bias.bias.copy_(torch.arange(model.relation_bias.num_types).float())
        biased = model(**inputs, candidate_relation_type_ids=relation_ids)
        model.relation_bias.bias.zero_()
        unbiased = model(**inputs, candidate_relation_type_ids=relation_ids)
    assert not torch.allclose(biased["base_table_logits"], unbiased["base_table_logits"])


def test_zero_relation_bias_does_not_change_output():
    model = _relation_model()
    inputs = _forward_inputs()
    with torch.no_grad():
        model.relation_bias.bias.zero_()
        first = model(**inputs, candidate_relation_type_ids=torch.zeros((1, 4, 4), dtype=torch.long))
        second = model(**inputs, candidate_relation_type_ids=torch.ones((1, 4, 4), dtype=torch.long))
    assert torch.allclose(first["base_table_logits"], second["base_table_logits"])


def test_nonzero_relation_bias_changes_output():
    test_relation_bias_changes_output_deterministically()


@pytest.mark.parametrize(("mode", "relation_fields", "expected"), [
    ("candidate_pairwise_relation_bias", {"candidate_relation_type_ids": torch.zeros((1, 4, 4), dtype=torch.long)}, "schema_candidate_pairwise_relation_bias"),
    ("schema_token_role_bias", {"relation_type_ids": torch.zeros((1, 3, 4), dtype=torch.long)}, "schema_token_role_bias"),
    ("schema_pairwise_relation_bias", {"schema_relation_type_ids": torch.zeros((1, 4, 4), dtype=torch.long)}, "schema_pairwise_relation_bias"),
    ("combined", {
        "relation_type_ids": torch.zeros((1, 3, 4), dtype=torch.long),
        "schema_relation_type_ids": torch.zeros((1, 4, 4), dtype=torch.long),
        "candidate_relation_type_ids": torch.zeros((1, 4, 4), dtype=torch.long),
    }, "combined"),
])
def test_relation_bias_mode_reports_active_paths(mode, relation_fields, expected):
    model = _relation_model(mode)
    with torch.no_grad():
        outputs = model(**_forward_inputs(), **relation_fields)
    assert outputs["relation_bias_mode"] == expected


def test_relation_bias_mode_reports_disabled():
    model = _relation_model(enabled=False)
    with torch.no_grad():
        outputs = model(**_forward_inputs())
    assert outputs["relation_bias_mode"] == "disabled"


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
    assert rat["candidate_relation_type_ids_configured"] is True
    assert rat["candidate_relation_type_ids_observed_in_batch"] is False
    assert rat["candidate_relation_type_ids_used_in_forward"] is False
    assert rat["candidate_pairwise_relation_bias_active"] is False


def test_relation_diagnostics_observe_dataset_batch_and_forward():
    diag = TrainingDiagnostics()
    diag.set_config({"model": {
        "max_tables": 2,
        "max_columns": 3,
        "relation_aware_attention": {
            "enabled": True,
            "relation_bias_mode": "candidate_pairwise_relation_bias",
        },
    }})
    diag.observe_dataset_item({"candidate_relation_type_ids": [[0]]})
    batch = {
        "candidate_relation_type_ids": torch.zeros((1, 5, 5), dtype=torch.long),
        "table_candidate_mask": torch.tensor([[1, 0]]),
        "column_candidate_mask": torch.tensor([[1, 1, 0]]),
    }
    diag.observe_step(batch, {
        "relation_bias_mode": "schema_candidate_pairwise_relation_bias",
        "candidate_pairwise_relation_bias_active": True,
        "candidate_level_relation_graph_available": True,
        "candidate_relation_type_ids_used_in_forward": True,
        "candidate_relation_attention_uses_mask": True,
    })
    rat = diag.config_summary["relation_aware_attention"]
    assert rat["candidate_relation_type_ids_observed_in_dataset"] is True
    assert rat["candidate_relation_type_ids_observed_in_batch"] is True
    assert rat["candidate_relation_type_ids_used_in_forward"] is True
    assert rat["candidate_relation_attention_uses_mask"] is True
    assert rat["relation_bias_mode"] == "schema_candidate_pairwise_relation_bias"
    graph = rat["candidate_relation_graph"]
    assert graph["actual_candidate_count_min"] == 3
    assert graph["actual_candidate_count_mean"] == 3
    assert graph["actual_candidate_count_max"] == 3
    assert graph["padded_candidate_count"] == 5
    assert graph["candidate_matrix_size"] == 25
    assert graph["padding_ratio_mean"] == pytest.approx(0.4)


def test_candidate_count_stats_aggregate_and_empty_masks_do_not_crash():
    diag = TrainingDiagnostics()
    diag.set_config({"model": {"max_tables": 2, "max_columns": 2}})
    diag.observe_batch({
        "table_candidate_mask": torch.tensor([[1, 0], [1, 1]]),
        "column_candidate_mask": torch.tensor([[1, 0], [1, 1]]),
    })
    graph = diag.config_summary["candidate_relation_graph"]
    assert graph["actual_candidate_count_min"] == 2
    assert graph["actual_candidate_count_mean"] == 3
    assert graph["actual_candidate_count_max"] == 4
    assert graph["padding_ratio_mean"] == pytest.approx(0.25)

    empty = TrainingDiagnostics()
    empty.set_config({"model": {"max_tables": 0, "max_columns": 0}})
    empty.observe_batch({
        "table_candidate_mask": torch.empty((1, 0)),
        "column_candidate_mask": torch.empty((1, 0)),
    })
    assert empty.config_summary["candidate_relation_graph"]["padding_ratio_mean"] == 0.0
