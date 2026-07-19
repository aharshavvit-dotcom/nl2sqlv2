"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import torch

from neural_ir.attention_model import SchemaAwareOptionAIRModel
from neural_ir.ir_label_encoder import IRLabelEncoder


def test_attention_model_forward_returns_logits_and_debug() -> None:
    encoder = IRLabelEncoder()
    model = SchemaAwareOptionAIRModel(
        config={"embedding_dim": 8, "hidden_dim": 8, "candidate_hidden_dim": 6, "max_tables": 3, "max_columns": 5, "max_candidate_tokens": 4},
        vocab_size=30,
        label_sizes=encoder.label_sizes,
    )
    outputs = model(
        question_ids=torch.ones((2, 6), dtype=torch.long),
        schema_ids=torch.ones((2, 10), dtype=torch.long),
        question_mask=torch.ones((2, 6)),
        schema_mask=torch.ones((2, 10)),
        table_candidate_token_ids=torch.ones((2, 3, 4), dtype=torch.long),
        column_candidate_token_ids=torch.ones((2, 5, 4), dtype=torch.long),
        table_candidate_mask=torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32),
        column_candidate_mask=torch.ones((2, 5)),
        metric_column_mask=torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.float32),
        schema_link_scores=torch.zeros((2, 5)),
    )

    for key in [
        "intent_logits",
        "base_table_logits",
        "metric_aggregation_logits",
        "metric_column_logits",
        "metric_expression_type_logits",
        "dimension_column_logits",
        "date_column_logits",
        "date_grain_logits",
        "date_filter_type_logits",
        "filter_column_logits",
        "filter_operator_logits",
        "order_direction_logits",
        "limit_bucket_logits",
    ]:
        assert key in outputs
    assert outputs["base_table_logits"][0, 1].item() <= -1e8
    assert outputs["metric_column_logits"][0, 1].item() <= -1e8
    assert "attention_weights" in outputs
    assert "candidate_scores" in outputs


def test_pointer_dropout_is_configured_and_active() -> None:
    encoder = IRLabelEncoder()
    model = SchemaAwareOptionAIRModel(
        config={"embedding_dim": 8, "hidden_dim": 8, "candidate_hidden_dim": 6, "pointer_dropout": 0.30},
        vocab_size=30,
        label_sizes=encoder.label_sizes,
    )

    assert isinstance(model.pointer_dropout, torch.nn.Dropout)
    assert model.pointer_dropout.p == 0.30
    model.train()
    values = torch.ones((128, 16))
    assert torch.count_nonzero(model.pointer_dropout(values) == 0) > 0


def test_pairwise_relation_bias_changes_logits() -> None:
    torch.manual_seed(7)
    encoder = IRLabelEncoder()
    base_config = {
        "embedding_dim": 8,
        "hidden_dim": 8,
        "candidate_hidden_dim": 6,
        "max_tables": 3,
        "max_columns": 5,
        "max_candidate_tokens": 4,
        "dropout": 0.0,
    }
    base = SchemaAwareOptionAIRModel(config=base_config, vocab_size=40, label_sizes=encoder.label_sizes).eval()
    related = SchemaAwareOptionAIRModel(
        config={
            **base_config,
            "relation_aware_attention": {
                "enabled": True,
                "relation_bias_mode": "schema_pairwise_relation_bias",
                "pairwise_relation_matrix": True,
                "bias_init": 0.0,
            },
        },
        vocab_size=40,
        label_sizes=encoder.label_sizes,
    ).eval()
    related.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        related.relation_bias.bias.copy_(torch.linspace(-1.0, 1.0, related.relation_bias.num_types))

    inputs = {
        "question_ids": torch.randint(1, 40, (1, 6)),
        "schema_ids": torch.randint(1, 40, (1, 10)),
        "question_mask": torch.ones((1, 6)),
        "schema_mask": torch.ones((1, 10)),
        "table_candidate_token_ids": torch.randint(1, 40, (1, 3, 4)),
        "column_candidate_token_ids": torch.randint(1, 40, (1, 5, 4)),
        "table_candidate_mask": torch.ones((1, 3)),
        "column_candidate_mask": torch.ones((1, 5)),
        "schema_link_scores": torch.zeros((1, 5)),
    }
    relation_ids = torch.arange(100, dtype=torch.long).reshape(1, 10, 10) % related.relation_bias.num_types

    base_logits = base(**inputs)["intent_logits"]
    related_outputs = related(**inputs, schema_relation_type_ids=relation_ids)

    assert related_outputs["relation_bias_mode"] == "schema_pairwise_relation_bias"
    assert not torch.allclose(base_logits, related_outputs["intent_logits"])
