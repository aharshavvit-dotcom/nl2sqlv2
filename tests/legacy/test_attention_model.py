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
