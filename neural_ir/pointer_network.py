from __future__ import annotations

import torch
from torch import nn


class SchemaPointerNetwork(nn.Module):
    def __init__(self, query_dim: int, candidate_dim: int, hidden_dim: int):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim, bias=False)
        self.candidate_projection = nn.Linear(candidate_dim, hidden_dim, bias=False)
        self.link_projection = nn.Linear(1, hidden_dim, bias=False)
        self.candidate_type_embedding = nn.Embedding(4, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_vector,
        candidate_vectors,
        candidate_mask=None,
        schema_link_scores=None,
    ):
        if candidate_vectors.dim() != 3:
            raise ValueError("candidate_vectors must have shape [batch, candidates, dim]")
        batch_size, candidate_count, _ = candidate_vectors.shape
        query = self.query_projection(query_vector).unsqueeze(1)
        candidates = self.candidate_projection(candidate_vectors)
        if schema_link_scores is None:
            link_scores = candidate_vectors.new_zeros((batch_size, candidate_count, 1))
        else:
            link_scores = schema_link_scores.to(candidate_vectors.device, dtype=candidate_vectors.dtype)
            if link_scores.dim() == 2:
                link_scores = link_scores.unsqueeze(-1)
            link_scores = _fit_last_dim(link_scores, candidate_count)
        type_ids = torch.zeros((batch_size, candidate_count), dtype=torch.long, device=candidate_vectors.device)
        type_bias = self.candidate_type_embedding(type_ids)
        hidden = torch.tanh(query + candidates + self.link_projection(link_scores) + type_bias)
        scores = self.scorer(hidden).squeeze(-1)
        if candidate_mask is not None:
            mask = candidate_mask.to(scores.device).bool()
            mask = _fit_last_dim(mask, candidate_count)
            scores = scores.masked_fill(~mask, -1e9)
        return scores


def _fit_last_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    current = tensor.size(1)
    if current == size:
        return tensor
    if current > size:
        return tensor[:, :size, ...]
    pad_shape = list(tensor.shape)
    pad_shape[1] = size - current
    pad = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad], dim=1)
