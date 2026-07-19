"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import torch

from neural_ir.pointer_network import SchemaPointerNetwork


def test_pointer_network_masks_invalid_candidates() -> None:
    pointer = SchemaPointerNetwork(query_dim=4, candidate_dim=5, hidden_dim=6)
    scores = pointer(
        query_vector=torch.ones((2, 4)),
        candidate_vectors=torch.ones((2, 3, 5)),
        candidate_mask=torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32),
    )

    assert scores.shape == (2, 3)
    assert scores[0, 1].item() <= -1e8
    assert scores[1, 0].item() <= -1e8
    assert scores[0, 0].item() > -1e8


def test_pointer_network_accepts_schema_link_scores() -> None:
    pointer = SchemaPointerNetwork(query_dim=4, candidate_dim=5, hidden_dim=6)
    scores = pointer(
        query_vector=torch.ones((1, 4)),
        candidate_vectors=torch.ones((1, 4, 5)),
        candidate_mask=torch.ones((1, 4)),
        schema_link_scores=torch.tensor([[0.1, 0.9, 0.0, 0.2]]),
    )

    assert scores.shape == (1, 4)
