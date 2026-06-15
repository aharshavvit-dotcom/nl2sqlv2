from __future__ import annotations

import torch
from torch import nn


DEFAULT_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 128,
    "dropout": 0.2,
    "max_question_len": 64,
    "max_schema_len": 256,
    "max_tables": 64,
    "max_columns": 256,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 0.001,
}


class OptionAIRModel(nn.Module):
    def __init__(self, config: dict, vocab_size: int, label_sizes: dict):
        super().__init__()
        merged = {**DEFAULT_CONFIG, **(config or {})}
        self.config = merged
        embedding_dim = int(merged["embedding_dim"])
        hidden_dim = int(merged["hidden_dim"])
        dropout = float(merged["dropout"])
        self.max_tables = int(merged["max_tables"])
        self.max_columns = int(merged["max_columns"])

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
        fused_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_dim = hidden_dim * 2
        self.intent_head = nn.Linear(head_dim, label_sizes["intent"])
        self.base_table_head = nn.Linear(head_dim, self.max_tables)
        self.metric_aggregation_head = nn.Linear(head_dim, label_sizes["metric_aggregation"])
        self.metric_column_head = nn.Linear(head_dim, self.max_columns)
        self.metric_expression_type_head = nn.Linear(head_dim, label_sizes["metric_expression_type"])
        self.dimension_column_head = nn.Linear(head_dim, self.max_columns)
        self.date_column_head = nn.Linear(head_dim, self.max_columns)
        self.date_grain_head = nn.Linear(head_dim, label_sizes["date_grain"])
        self.date_filter_type_head = nn.Linear(head_dim, label_sizes["date_filter_type"])
        self.filter_column_head = nn.Linear(head_dim, self.max_columns)
        self.filter_operator_head = nn.Linear(head_dim, label_sizes["filter_operator"])
        self.order_direction_head = nn.Linear(head_dim, label_sizes["order_direction"])
        self.limit_bucket_head = nn.Linear(head_dim, label_sizes["limit_bucket"])
        self.metric_link_scale = nn.Parameter(torch.tensor(0.5))
        self.dimension_link_scale = nn.Parameter(torch.tensor(0.5))
        self.date_link_scale = nn.Parameter(torch.tensor(0.5))
        self.filter_link_scale = nn.Parameter(torch.tensor(0.5))

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
    ) -> dict[str, torch.Tensor]:
        question_emb = self.embedding(question_ids)
        schema_emb = self.embedding(schema_ids)
        question_out, _ = self.question_encoder(question_emb)
        schema_out, _ = self.schema_encoder(schema_emb)
        question_vec = self._masked_mean(question_out, question_mask)
        schema_vec = self._masked_mean(schema_out, schema_mask)
        fused = self.fusion(torch.cat([question_vec, schema_vec], dim=-1))
        base_table_logits = self.base_table_head(fused)
        metric_column_logits = self._add_link_scores(self.metric_column_head(fused), schema_link_scores, self.metric_link_scale)
        dimension_column_logits = self._add_link_scores(self.dimension_column_head(fused), schema_link_scores, self.dimension_link_scale)
        date_column_logits = self._add_link_scores(self.date_column_head(fused), schema_link_scores, self.date_link_scale)
        filter_column_logits = self._add_link_scores(self.filter_column_head(fused), schema_link_scores, self.filter_link_scale)
        return {
            "intent_logits": self.intent_head(fused),
            "base_table_logits": masked_logits(base_table_logits, table_candidate_mask),
            "metric_aggregation_logits": self.metric_aggregation_head(fused),
            "metric_column_logits": masked_logits(metric_column_logits, _preferred_mask(metric_column_mask, column_candidate_mask)),
            "metric_expression_type_logits": self.metric_expression_type_head(fused),
            "dimension_column_logits": masked_logits(dimension_column_logits, _preferred_mask(dimension_column_mask, column_candidate_mask)),
            "date_column_logits": masked_logits(date_column_logits, _preferred_mask(date_column_mask, column_candidate_mask)),
            "date_grain_logits": self.date_grain_head(fused),
            "date_filter_type_logits": self.date_filter_type_head(fused),
            "filter_column_logits": masked_logits(filter_column_logits, _preferred_mask(filter_column_mask, column_candidate_mask)),
            "filter_operator_logits": self.filter_operator_head(fused),
            "order_direction_logits": self.order_direction_head(fused),
            "limit_bucket_logits": self.limit_bucket_head(fused),
        }

    @staticmethod
    def _masked_mean(outputs: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return outputs.mean(dim=1)
        weights = mask.unsqueeze(-1).to(outputs.dtype)
        total = (outputs * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return total / denom

    @staticmethod
    def _add_link_scores(logits: torch.Tensor, schema_link_scores: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
        if schema_link_scores is None:
            return logits
        scores = schema_link_scores.to(logits.device, dtype=logits.dtype)
        if scores.size(-1) != logits.size(-1):
            scores = scores[..., : logits.size(-1)]
            if scores.size(-1) < logits.size(-1):
                scores = torch.nn.functional.pad(scores, (0, logits.size(-1) - scores.size(-1)))
        return logits + scores * scale


def masked_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask.to(logits.device).bool(), -1e9)


def _preferred_mask(primary: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
    return primary if primary is not None else fallback
