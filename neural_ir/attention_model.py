from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from neural_optimization.activation_factory import get_activation

from .model import masked_logits
from .pointer_network import SchemaPointerNetwork


def _maybe_ffn_head(input_dim: int, output_dim: int, config: dict) -> nn.Module:
    """Return an FFN-wrapped head or a plain Linear, depending on config."""
    if not config.get("feed_forward_heads", False):
        return nn.Linear(input_dim, output_dim)
    try:
        from neural_optimization.ffn_blocks import FeedForwardBlock
    except ImportError:
        return nn.Linear(input_dim, output_dim)
    ffn_dim = int(config.get("feed_forward_dim", 256))
    activation = str(config.get("activation", "gelu"))
    dropout = float(config.get("dropout", 0.25))
    layer_norm = bool(config.get("layer_norm", True))
    return nn.Sequential(
        FeedForwardBlock(
            input_dim=input_dim,
            hidden_dim=ffn_dim,
            output_dim=input_dim,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            residual=True,
        ),
        nn.Linear(input_dim, output_dim),
    )


DEFAULT_V2_CONFIG = {
    "model_version": "schema_aware_queryir_v1",
    "embedding_dim": 128,
    "hidden_dim": 192,
    "candidate_hidden_dim": 128,
    "dropout": 0.25,
    "max_question_len": 64,
    "max_schema_len": 320,
    "max_candidate_tokens": 16,
    "max_tables": 64,
    "max_columns": 256,
    "batch_size": 8,
    "epochs": 8,
    "learning_rate": 0.0007,
    "weight_decay": 0.00001,
    "use_hard_negative_loss": True,
    "hard_negative_loss_weight": 0.3,
}


class SchemaAwareOptionAIRModel(nn.Module):
    def __init__(self, config: dict, vocab_size: int, label_sizes: dict):
        super().__init__()
        merged = {**DEFAULT_V2_CONFIG, **(config or {})}
        self.config = merged
        embedding_dim = int(merged["embedding_dim"])
        hidden_dim = int(merged["hidden_dim"])
        candidate_hidden_dim = int(merged["candidate_hidden_dim"])
        dropout = float(merged["dropout"])
        self.max_tables = int(merged["max_tables"])
        self.max_columns = int(merged["max_columns"])
        self.max_candidate_tokens = int(merged.get("max_candidate_tokens", 16))

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.question_encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.schema_encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.candidate_encoder = nn.GRU(
            embedding_dim,
            candidate_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        sequence_dim = hidden_dim * 2
        candidate_dim = candidate_hidden_dim * 2
        head_dim = sequence_dim
        self.fusion = nn.Sequential(
            nn.Linear(sequence_dim * 4, head_dim),
            get_activation(str(merged.get("activation", "gelu"))),
            nn.Dropout(dropout),
        )
        self.fallback_candidate_projection = nn.Linear(sequence_dim, candidate_dim)

        self.intent_head = _maybe_ffn_head(head_dim, label_sizes["intent"], merged)
        self.metric_aggregation_head = _maybe_ffn_head(head_dim, label_sizes["metric_aggregation"], merged)
        self.metric_expression_type_head = _maybe_ffn_head(head_dim, label_sizes["metric_expression_type"], merged)
        self.date_grain_head = _maybe_ffn_head(head_dim, label_sizes["date_grain"], merged)
        self.date_filter_type_head = _maybe_ffn_head(head_dim, label_sizes["date_filter_type"], merged)
        self.filter_operator_head = _maybe_ffn_head(head_dim, label_sizes["filter_operator"], merged)
        self.order_direction_head = _maybe_ffn_head(head_dim, label_sizes["order_direction"], merged)
        self.limit_bucket_head = _maybe_ffn_head(head_dim, label_sizes["limit_bucket"], merged)

        self.table_pointer = SchemaPointerNetwork(head_dim, candidate_dim, candidate_hidden_dim)
        self.metric_pointer = SchemaPointerNetwork(head_dim, candidate_dim, candidate_hidden_dim)
        self.dimension_pointer = SchemaPointerNetwork(head_dim, candidate_dim, candidate_hidden_dim)
        self.date_pointer = SchemaPointerNetwork(head_dim, candidate_dim, candidate_hidden_dim)
        self.filter_pointer = SchemaPointerNetwork(head_dim, candidate_dim, candidate_hidden_dim)

    def forward(
        self,
        question_ids,
        schema_ids,
        candidate_token_ids=None,
        question_mask=None,
        schema_mask=None,
        table_candidate_mask=None,
        column_candidate_mask=None,
        metric_column_mask=None,
        dimension_column_mask=None,
        date_column_mask=None,
        filter_column_mask=None,
        schema_link_scores=None,
        table_candidate_token_ids=None,
        column_candidate_token_ids=None,
    ) -> dict:
        question_emb = self.embedding(question_ids)
        schema_emb = self.embedding(schema_ids)
        question_out, _ = self.question_encoder(question_emb)
        schema_out, _ = self.schema_encoder(schema_emb)

        question_vec = self._masked_mean(question_out, question_mask)
        schema_vec = self._masked_mean(schema_out, schema_mask)
        attended_schema, attention_weights = self._schema_attention(
            question_out,
            schema_out,
            question_mask=question_mask,
            schema_mask=schema_mask,
        )
        fused = self.fusion(torch.cat([question_vec, schema_vec, attended_schema, question_vec * attended_schema], dim=-1))

        if column_candidate_token_ids is None:
            column_candidate_token_ids = candidate_token_ids
        table_vectors = self._candidate_vectors(table_candidate_token_ids, self.max_tables, schema_vec)
        column_vectors = self._candidate_vectors(column_candidate_token_ids, self.max_columns, schema_vec)
        table_link_scores = schema_link_scores.new_zeros((schema_link_scores.size(0), self.max_tables)) if schema_link_scores is not None else None

        base_table_logits = self.table_pointer(fused, table_vectors, table_candidate_mask, table_link_scores)
        metric_column_logits = self.metric_pointer(
            fused,
            column_vectors,
            _preferred_mask(metric_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        dimension_column_logits = self.dimension_pointer(
            fused,
            column_vectors,
            _preferred_mask(dimension_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        date_column_logits = self.date_pointer(
            fused,
            column_vectors,
            _preferred_mask(date_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        filter_column_logits = self.filter_pointer(
            fused,
            column_vectors,
            _preferred_mask(filter_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        return {
            "intent_logits": self.intent_head(fused),
            "base_table_logits": masked_logits(base_table_logits, table_candidate_mask),
            "metric_aggregation_logits": self.metric_aggregation_head(fused),
            "metric_column_logits": metric_column_logits,
            "metric_expression_type_logits": self.metric_expression_type_head(fused),
            "dimension_column_logits": dimension_column_logits,
            "date_column_logits": date_column_logits,
            "date_grain_logits": self.date_grain_head(fused),
            "date_filter_type_logits": self.date_filter_type_head(fused),
            "filter_column_logits": filter_column_logits,
            "filter_operator_logits": self.filter_operator_head(fused),
            "order_direction_logits": self.order_direction_head(fused),
            "limit_bucket_logits": self.limit_bucket_head(fused),
            "attention_weights": attention_weights.detach(),
            "top_schema_candidates": self._top_schema_candidates(attention_weights, schema_mask),
            "candidate_scores": {
                "base_table": base_table_logits.detach(),
                "metric_column": metric_column_logits.detach(),
                "dimension_column": dimension_column_logits.detach(),
                "date_column": date_column_logits.detach(),
                "filter_column": filter_column_logits.detach(),
            },
        }

    def _schema_attention(
        self,
        question_out: torch.Tensor,
        schema_out: torch.Tensor,
        question_mask: torch.Tensor | None,
        schema_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = math.sqrt(max(question_out.size(-1), 1))
        scores = torch.matmul(question_out, schema_out.transpose(1, 2)) / scale
        if schema_mask is not None:
            scores = scores.masked_fill(~schema_mask.to(scores.device).bool().unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, schema_out)
        return self._masked_mean(context, question_mask), weights

    def _candidate_vectors(
        self,
        token_ids: torch.Tensor | None,
        max_candidates: int,
        fallback_schema_vec: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids is None:
            fallback = self.fallback_candidate_projection(fallback_schema_vec)
            return fallback.unsqueeze(1).expand(-1, max_candidates, -1).contiguous()
        token_ids = token_ids.to(fallback_schema_vec.device)
        if token_ids.dim() != 3:
            raise ValueError("candidate token ids must have shape [batch, candidates, tokens]")
        token_ids = _fit_candidate_count(token_ids, max_candidates)
        batch_size, candidate_count, token_count = token_ids.shape
        flat = token_ids.reshape(batch_size * candidate_count, token_count)
        active = flat.ne(0).any(dim=1)
        pooled = self.fallback_candidate_projection(fallback_schema_vec).new_zeros((batch_size * candidate_count, self.config["candidate_hidden_dim"] * 2))
        if active.any():
            active_flat = flat[active]
            embedded = self.embedding(active_flat)
            encoded, _ = self.candidate_encoder(embedded)
            mask = active_flat.ne(0).float()
            pooled[active] = self._masked_mean(encoded, mask)
        return pooled.reshape(batch_size, candidate_count, -1)

    @staticmethod
    def _masked_mean(outputs: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return outputs.mean(dim=1)
        weights = mask.unsqueeze(-1).to(outputs.device, dtype=outputs.dtype)
        total = (outputs * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return total / denom

    @staticmethod
    def _top_schema_candidates(attention_weights: torch.Tensor, schema_mask: torch.Tensor | None) -> torch.Tensor:
        scores = attention_weights.mean(dim=1)
        if schema_mask is not None:
            scores = scores.masked_fill(~schema_mask.to(scores.device).bool(), -1e9)
        top_k = min(5, scores.size(-1))
        return torch.topk(scores, k=top_k, dim=-1).indices.detach()


def _preferred_mask(primary: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
    return primary if primary is not None else fallback


def _fit_candidate_count(token_ids: torch.Tensor, size: int) -> torch.Tensor:
    current = token_ids.size(1)
    if current == size:
        return token_ids
    if current > size:
        return token_ids[:, :size, :]
    pad = token_ids.new_zeros((token_ids.size(0), size - current, token_ids.size(2)))
    return torch.cat([token_ids, pad], dim=1)
