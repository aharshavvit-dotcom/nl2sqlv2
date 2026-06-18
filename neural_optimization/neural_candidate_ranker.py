"""Neural candidate ranker — lightweight FFN for ranking QueryIR candidates.

Scores each candidate with a sigmoid output indicating quality.
Uses hand-crafted features (retrieval score, neural confidence, schema
overlap, validation flags, etc.) rather than raw text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .activation_factory import get_activation


DEFAULT_RANKER_FEATURES = [
    "retrieval_score",
    "neural_confidence",
    "generic_planner_confidence",
    "schema_overlap_score",
    "intent_pattern_score",
    "semantic_profile_score",
    "queryir_validation_passed",
    "sql_validation_passed",
    "join_count",
    "join_policy_respected",
    "has_filter_when_requested",
    "has_group_by_when_requested",
    "hard_negative_similarity",
    "unnecessary_join_flag",
    "wrong_table_risk",
]


class NeuralCandidateRanker(nn.Module):
    """FFN ranker: feature vector → hidden → sigmoid score.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        Hidden layer width.
    activation:
        Activation name for hidden layer.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return sigmoid scores in [0, 1]."""
        return torch.sigmoid(self.net(features)).squeeze(-1)


def save_ranker(
    model: NeuralCandidateRanker,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    training_report: dict[str, Any] | None = None,
) -> None:
    """Persist ranker model, config, and training report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "ranker.pt")
    (out / "ranker_config.json").write_text(
        json.dumps(config or {}, indent=2), encoding="utf-8",
    )
    if training_report:
        (out / "ranker_training_report.json").write_text(
            json.dumps(training_report, indent=2, default=str), encoding="utf-8",
        )


def load_ranker(model_dir: str | Path) -> NeuralCandidateRanker:
    """Load a persisted ranker."""
    d = Path(model_dir)
    cfg = json.loads((d / "ranker_config.json").read_text(encoding="utf-8")) if (d / "ranker_config.json").exists() else {}
    ranker = NeuralCandidateRanker(
        input_dim=cfg.get("input_dim", len(DEFAULT_RANKER_FEATURES)),
        hidden_dim=cfg.get("hidden_dim", 64),
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.2),
    )
    state = torch.load(d / "ranker.pt", map_location="cpu", weights_only=False)
    ranker.load_state_dict(state)
    ranker.eval()
    return ranker
