"""Gradient connectivity test for NL2SQL neural model.

Verifies that every active loss head produces non-zero gradients
on a synthetic batch with valid labels. This is the definitive
proof that the model -> loss -> gradient chain is intact for
each training signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _build_model_and_batch():
    """Build a small model and synthetic batch with valid labels for all heads."""
    from neural_ir.attention_model import SchemaAwareOptionAIRModel

    # Small config for fast testing
    config = {
        "embedding_dim": 32,
        "hidden_dim": 16,
        "candidate_hidden_dim": 16,
        "dropout": 0.0,
        "max_tables": 4,
        "max_columns": 8,
        "max_candidate_tokens": 4,
        "activation": "gelu",
        "pointer_dropout": 0.0,
        "feed_forward_heads": False,
        "relation_aware_attention": {"enabled": False},
    }

    label_sizes = {
        "intent": 5,
        "metric_aggregation": 4,
        "metric_expression_type": 3,
        "date_grain": 4,
        "date_filter_type": 3,
        "filter_operator": 6,
        "order_direction": 3,
        "limit_bucket": 5,
    }

    vocab_size = 100
    model = SchemaAwareOptionAIRModel(config, vocab_size, label_sizes)

    batch_size = 2
    max_q = 16
    max_s = 32
    max_tables = config["max_tables"]
    max_columns = config["max_columns"]
    max_ct = config["max_candidate_tokens"]

    batch = {
        "question_ids": torch.randint(1, vocab_size, (batch_size, max_q)),
        "schema_ids": torch.randint(1, vocab_size, (batch_size, max_s)),
        "question_mask": torch.ones(batch_size, max_q),
        "schema_mask": torch.ones(batch_size, max_s),
        "table_candidate_mask": torch.ones(batch_size, max_tables),
        "column_candidate_mask": torch.ones(batch_size, max_columns),
        "metric_column_mask": torch.ones(batch_size, max_columns),
        "dimension_column_mask": torch.ones(batch_size, max_columns),
        "date_column_mask": torch.ones(batch_size, max_columns),
        "filter_column_mask": torch.ones(batch_size, max_columns),
        "schema_link_scores": torch.randn(batch_size, max_columns),
        "table_candidate_token_ids": torch.randint(1, vocab_size, (batch_size, max_tables, max_ct)),
        "column_candidate_token_ids": torch.randint(1, vocab_size, (batch_size, max_columns, max_ct)),
        "candidate_token_ids": torch.randint(1, vocab_size, (batch_size, max_columns, max_ct)),
    }

    # Labels: valid non-negative indices for all heads
    labels = {
        "intent_label": torch.tensor([0, 1]),
        "metric_aggregation_label": torch.tensor([0, 1]),
        "metric_expression_type_label": torch.tensor([0, 1]),
        "date_grain_label": torch.tensor([0, 1]),
        "date_filter_type_label": torch.tensor([0, 1]),
        "filter_operator_label": torch.tensor([0, 1]),
        "order_direction_label": torch.tensor([0, 1]),
        "limit_bucket_label": torch.tensor([0, 1]),
        "base_table_index": torch.tensor([0, 1]),
        "metric_column_index": torch.tensor([0, 1]),
        "dimension_column_index": torch.tensor([0, 1]),
        "date_column_index": torch.tensor([0, 1]),
        "filter_column_index": torch.tensor([0, 1]),
    }

    return model, batch, labels, label_sizes


# Map from output head name to label key
HEAD_TO_LABEL = {
    "intent_logits": "intent_label",
    "metric_aggregation_logits": "metric_aggregation_label",
    "metric_expression_type_logits": "metric_expression_type_label",
    "date_grain_logits": "date_grain_label",
    "date_filter_type_logits": "date_filter_type_label",
    "filter_operator_logits": "filter_operator_label",
    "order_direction_logits": "order_direction_label",
    "limit_bucket_logits": "limit_bucket_label",
    "base_table_logits": "base_table_index",
    "metric_column_logits": "metric_column_index",
    "dimension_column_logits": "dimension_column_index",
    "date_column_logits": "date_column_index",
    "filter_column_logits": "filter_column_index",
}


class TestGradientConnectivity:
    """Verify every head receives non-zero gradient on valid input."""

    @pytest.fixture(scope="class")
    def model_and_batch(self):
        return _build_model_and_batch()

    @pytest.mark.parametrize("head_name,label_key", list(HEAD_TO_LABEL.items()))
    def test_head_produces_nonzero_gradient(self, model_and_batch, head_name, label_key):
        """Each head produces non-zero loss and gradient when given valid labels."""
        model, batch, labels, label_sizes = model_and_batch

        model.zero_grad()
        model.train()

        # Forward pass
        forward_kwargs = {
            k: batch[k] for k in [
                "question_ids", "schema_ids", "question_mask", "schema_mask",
                "table_candidate_mask", "column_candidate_mask",
                "metric_column_mask", "dimension_column_mask",
                "date_column_mask", "filter_column_mask",
                "schema_link_scores",
                "table_candidate_token_ids", "column_candidate_token_ids",
            ]
        }
        outputs = model(**forward_kwargs)

        assert head_name in outputs, f"Head {head_name} not in model outputs: {list(outputs.keys())}"

        logits = outputs[head_name]
        target = labels[label_key]

        # Compute loss for this head
        if "index" in label_key:
            # Pointer head: cross entropy with logits
            loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1)
        else:
            # Classification head
            loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1)

        assert loss.item() > 0, f"Loss for {head_name} is zero with valid labels"

        loss.backward()

        # Check that at least some parameters have non-zero gradient
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                has_grad = True
                break

        assert has_grad, (
            f"Head {head_name} produced loss={loss.item():.6f} but no parameter "
            f"received non-zero gradient. The gradient chain is broken."
        )

    def test_all_heads_present_in_output(self, model_and_batch):
        """All expected heads are present in model output."""
        model, batch, labels, label_sizes = model_and_batch

        model.eval()
        forward_kwargs = {
            k: batch[k] for k in [
                "question_ids", "schema_ids", "question_mask", "schema_mask",
                "table_candidate_mask", "column_candidate_mask",
                "metric_column_mask", "dimension_column_mask",
                "date_column_mask", "filter_column_mask",
                "schema_link_scores",
                "table_candidate_token_ids", "column_candidate_token_ids",
            ]
        }
        with torch.no_grad():
            outputs = model(**forward_kwargs)

        for head_name in HEAD_TO_LABEL:
            assert head_name in outputs, f"Missing head: {head_name}"
