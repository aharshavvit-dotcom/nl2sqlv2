"""Ranker dataset builder.

Converts gold comparison results into (feature_vector, label) pairs
for training the :class:`NeuralCandidateRanker`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .neural_candidate_ranker import DEFAULT_RANKER_FEATURES


class RankerDataset(Dataset):
    """PyTorch dataset of (feature_vector, label) for candidate ranking."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        features = torch.tensor(
            [float(ex.get(f, 0.0)) for f in DEFAULT_RANKER_FEATURES],
            dtype=torch.float32,
        )
        label = torch.tensor(float(ex.get("label", 0.0)), dtype=torch.float32)
        return {"features": features, "label": label}


def build_ranker_dataset(
    examples: list[dict[str, Any]],
) -> RankerDataset:
    """Build a ``RankerDataset`` from raw example dicts."""
    return RankerDataset(examples)


def load_ranker_examples(path: str | Path) -> list[dict[str, Any]]:
    """Load ranking examples from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
