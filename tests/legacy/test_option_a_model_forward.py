"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import torch

from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.model import OptionAIRModel


def test_option_a_model_forward_returns_required_logits() -> None:
    encoder = IRLabelEncoder()
    model = OptionAIRModel(
        config={"embedding_dim": 8, "hidden_dim": 8, "max_tables": 4, "max_columns": 8},
        vocab_size=20,
        label_sizes=encoder.label_sizes,
    )
    outputs = model(
        question_ids=torch.ones((2, 6), dtype=torch.long),
        schema_ids=torch.ones((2, 10), dtype=torch.long),
        question_mask=torch.ones((2, 6)),
        schema_mask=torch.ones((2, 10)),
    )

    assert set(outputs) == {
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
    }
    assert outputs["intent_logits"].shape[0] == 2
    assert outputs["metric_column_logits"].shape[1] == 8
