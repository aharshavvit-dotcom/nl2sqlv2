from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from neural_optimization.activation_factory import get_activation

from .model import masked_logits
from .pointer_network import SchemaPointerNetwork

# Relation types for relation-aware schema attention (RAT-SQL-style bias)
RELATION_TYPES = [
    "same_table", "table_has_column", "column_belongs_to_table",
    "fk_to_pk", "pk_to_fk", "primary_key", "foreign_key_column",
    "same_column_name", "same_data_type", "unrelated",
]


class RelationBiasModule(nn.Module):
    """Learnable scalar bias per schema relation type.

    Adds relation-dependent bias to schema attention logits before softmax.
    This is a lightweight RAT-SQL-style relation-aware attention mechanism.
    """

    def __init__(self, num_types: int = len(RELATION_TYPES), bias_init: float = 0.0):
        super().__init__()
        self.num_types = num_types
        self.bias = nn.Parameter(torch.full((num_types,), bias_init))

    def forward(self, relation_type_ids: torch.Tensor) -> torch.Tensor:
        """Return bias values for given relation type IDs.

        Args:
            relation_type_ids: [batch, query_len, schema_len] of int type indices.

        Returns:
            [batch, query_len, schema_len] bias to add to attention scores.
        """
        return self.bias[relation_type_ids.clamp(0, self.num_types - 1)]


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
    "epochs": 10,
    "learning_rate": 0.0007,
    "weight_decay": 0.00001,
    "use_hard_negative_loss": True,
    "hard_negative_loss_weight": 0.3,
    "relation_aware_attention": {
        "enabled": False,
        "relation_bias_mode": "schema_pairwise_relation_bias",
        "pairwise_relation_matrix": True,
        "relation_types": RELATION_TYPES,
        "bias_init": 0.0,
    },
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

        # Relation-aware schema attention (experimental, disabled by default)
        rat_config = merged.get("relation_aware_attention") or {}
        self.relation_aware_enabled = bool(rat_config.get("enabled", False))
        self.pairwise_relation_matrix_enabled = bool(rat_config.get("pairwise_relation_matrix", True))
        self.relation_bias_mode = str(
            rat_config.get("relation_bias_mode")
            or (
                "schema_pairwise_relation_bias"
                if self.pairwise_relation_matrix_enabled
                else "schema_token_role_bias"
            )
        )
        if self.relation_aware_enabled:
            relation_types = rat_config.get("relation_types", RELATION_TYPES)
            bias_init = float(rat_config.get("bias_init", 0.0))
            self.relation_bias = RelationBiasModule(
                num_types=len(relation_types), bias_init=bias_init,
            )
        else:
            self.relation_bias = None

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
        self.pointer_dropout = nn.Dropout(float(merged.get("pointer_dropout", 0.30)))

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
        relation_type_ids=None,
        schema_relation_type_ids=None,
        candidate_relation_type_ids=None,
    ) -> dict:
        question_emb = self.embedding(question_ids)
        schema_emb = self.embedding(schema_ids)
        question_out, _ = self.question_encoder(question_emb)
        schema_out, _ = self.schema_encoder(schema_emb)

        # Track active relation paths independently.  The effective mode is
        # computed only after every path has had a chance to run.
        schema_pairwise_active = False
        schema_role_active = False
        candidate_pairwise_active = False

        if (
            self.relation_bias is not None
            and schema_relation_type_ids is not None
            and self.relation_bias_mode in ("schema_pairwise_relation_bias", "combined")
        ):
            schema_out = self._schema_pairwise_relation_context(
                schema_out,
                schema_mask=schema_mask,
                schema_relation_type_ids=schema_relation_type_ids,
            )
            schema_pairwise_active = True

        question_vec = self._masked_mean(question_out, question_mask)
        schema_vec = self._masked_mean(schema_out, schema_mask)

        # Pass relation_type_ids if role bias is enabled (either alone or combined)
        role_bias_ids = None
        if self.relation_bias is not None and self.relation_bias_mode in ("schema_token_role_bias", "combined"):
            role_bias_ids = relation_type_ids
            schema_role_active = role_bias_ids is not None

        attended_schema, attention_weights = self._schema_attention(
            question_out,
            schema_out,
            question_mask=question_mask,
            schema_mask=schema_mask,
            relation_type_ids=role_bias_ids,
        )

        fused = self.fusion(torch.cat([question_vec, schema_vec, attended_schema, question_vec * attended_schema], dim=-1))

        if column_candidate_token_ids is None:
            column_candidate_token_ids = candidate_token_ids
        table_vectors = self._candidate_vectors(table_candidate_token_ids, self.max_tables, schema_vec)
        column_vectors = self._candidate_vectors(column_candidate_token_ids, self.max_columns, schema_vec)
        
        # Apply candidate-level pairwise relation context without allowing
        # padded candidates to affect the attention normalization.
        if (
            self.relation_bias is not None
            and candidate_relation_type_ids is not None
            and self.relation_bias_mode in (
                "candidate_pairwise_relation_bias",
                "schema_candidate_pairwise_relation_bias",
                "combined",
            )
        ):
            candidate_mask = self._unified_candidate_mask(
                table_candidate_mask,
                column_candidate_mask,
                batch_size=table_vectors.size(0),
                device=table_vectors.device,
            )
            unified_candidate_vectors = torch.cat([table_vectors, column_vectors], dim=1)
            unified_candidate_vectors = self._candidate_pairwise_relation_context(
                unified_candidate_vectors,
                candidate_mask=candidate_mask,
                candidate_relation_type_ids=candidate_relation_type_ids,
            )
            candidate_pairwise_active = True
            table_vectors = unified_candidate_vectors[:, :self.max_tables, :]
            column_vectors = unified_candidate_vectors[:, self.max_tables:, :]

        active_paths = sum((schema_pairwise_active, schema_role_active, candidate_pairwise_active))
        if active_paths > 1:
            active_relation_bias_mode = "combined"
        elif candidate_pairwise_active:
            active_relation_bias_mode = "schema_candidate_pairwise_relation_bias"
        elif schema_pairwise_active:
            active_relation_bias_mode = "schema_pairwise_relation_bias"
        elif schema_role_active:
            active_relation_bias_mode = "schema_token_role_bias"
        else:
            active_relation_bias_mode = "disabled"
            
        table_link_scores = schema_link_scores.new_zeros((schema_link_scores.size(0), self.max_tables)) if schema_link_scores is not None else None

        pointer_input = self.pointer_dropout(fused)
        base_table_logits = self.table_pointer(pointer_input, table_vectors, table_candidate_mask, table_link_scores)
        metric_column_logits = self.metric_pointer(
            pointer_input,
            column_vectors,
            _preferred_mask(metric_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        dimension_column_logits = self.dimension_pointer(
            pointer_input,
            column_vectors,
            _preferred_mask(dimension_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        date_column_logits = self.date_pointer(
            pointer_input,
            column_vectors,
            _preferred_mask(date_column_mask, column_candidate_mask),
            schema_link_scores,
        )
        filter_column_logits = self.filter_pointer(
            pointer_input,
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
            "relation_bias_mode": active_relation_bias_mode,
            "question_schema_role_bias_active": schema_role_active,
            "schema_pairwise_relation_bias_active": schema_pairwise_active,
            "candidate_pairwise_relation_bias_active": candidate_pairwise_active,
            "candidate_level_relation_graph_available": candidate_relation_type_ids is not None,
            "candidate_relation_type_ids_used_in_forward": candidate_pairwise_active,
            "schema_relation_type_ids_used_in_forward": schema_pairwise_active,
            "relation_type_ids_used_in_forward": schema_role_active,
            "candidate_relation_attention_uses_mask": candidate_pairwise_active,
        }

    def _schema_attention(
        self,
        question_out: torch.Tensor,
        schema_out: torch.Tensor,
        question_mask: torch.Tensor | None,
        schema_mask: torch.Tensor | None,
        relation_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = math.sqrt(max(question_out.size(-1), 1))
        scores = torch.matmul(question_out, schema_out.transpose(1, 2)) / scale
        # Relation-aware bias: add learned per-relation-type scalar bias
        if self.relation_bias is not None and relation_type_ids is not None:
            relation_bias = self.relation_bias(relation_type_ids.to(scores.device))
            scores = scores + relation_bias
        if schema_mask is not None:
            scores = scores.masked_fill(~schema_mask.to(scores.device).bool().unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, schema_out)
        return self._masked_mean(context, question_mask), weights

    def _schema_pairwise_relation_context(
        self,
        schema_out: torch.Tensor,
        schema_mask: torch.Tensor | None,
        schema_relation_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        scale = math.sqrt(max(schema_out.size(-1), 1))
        scores = torch.matmul(schema_out, schema_out.transpose(1, 2)) / scale
        relation_bias = self.relation_bias(schema_relation_type_ids.to(scores.device))
        scores = scores + relation_bias
        if schema_mask is not None:
            mask = schema_mask.to(scores.device).bool()
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, schema_out)
        return schema_out + context

    def _candidate_pairwise_relation_context(
        self,
        candidate_vectors: torch.Tensor,
        candidate_mask: torch.Tensor | None,
        candidate_relation_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_vectors.dim() != 3:
            raise ValueError("candidate_vectors must have shape [batch, candidates, hidden]")
        batch_size, candidate_count, _ = candidate_vectors.shape
        expected_relation_shape = (batch_size, candidate_count, candidate_count)
        if tuple(candidate_relation_type_ids.shape) != expected_relation_shape:
            raise ValueError(
                "candidate_relation_type_ids must have shape "
                f"{expected_relation_shape}, got {tuple(candidate_relation_type_ids.shape)}"
            )
        if candidate_mask is None:
            raise ValueError("candidate_mask is required for candidate-pairwise relation attention")
        expected_mask_shape = (batch_size, candidate_count)
        if tuple(candidate_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"candidate_mask must have shape {expected_mask_shape}, got {tuple(candidate_mask.shape)}"
            )

        scale = math.sqrt(max(candidate_vectors.size(-1), 1))
        scores = torch.matmul(candidate_vectors, candidate_vectors.transpose(1, 2)) / scale
        relation_bias = self.relation_bias(candidate_relation_type_ids.to(scores.device))
        scores = scores + relation_bias
        mask = candidate_mask.to(scores.device).bool()
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, candidate_vectors)
        return (candidate_vectors + context) * mask.unsqueeze(-1).to(candidate_vectors.dtype)

    def _unified_candidate_mask(
        self,
        table_candidate_mask: torch.Tensor | None,
        column_candidate_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if table_candidate_mask is None or column_candidate_mask is None:
            raise ValueError(
                "table_candidate_mask and column_candidate_mask are required "
                "for candidate-pairwise relation attention"
            )
        expected_table_shape = (batch_size, self.max_tables)
        expected_column_shape = (batch_size, self.max_columns)
        if tuple(table_candidate_mask.shape) != expected_table_shape:
            raise ValueError(
                f"table_candidate_mask must have shape {expected_table_shape}, "
                f"got {tuple(table_candidate_mask.shape)}"
            )
        if tuple(column_candidate_mask.shape) != expected_column_shape:
            raise ValueError(
                f"column_candidate_mask must have shape {expected_column_shape}, "
                f"got {tuple(column_candidate_mask.shape)}"
            )
        return torch.cat(
            [table_candidate_mask.to(device), column_candidate_mask.to(device)],
            dim=1,
        ).bool()

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
