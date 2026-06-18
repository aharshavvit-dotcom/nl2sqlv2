"""CLI for training the Neural Candidate Ranker.

Usage:
    python training/train_neural_candidate_ranker.py \\
      --ranking-data data/processed/self_training/ranking_examples.jsonl \\
      --output-dir artifacts/neural_candidate_ranker
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from neural_optimization.neural_candidate_ranker import (
    NeuralCandidateRanker,
    DEFAULT_RANKER_FEATURES,
    save_ranker,
)
from neural_optimization.ranker_dataset_builder import build_ranker_dataset, load_ranker_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Neural Candidate Ranker")
    parser.add_argument("--ranking-data", type=str, default="data/processed/self_training/ranking_examples.jsonl")
    parser.add_argument("--output-dir", type=str, default="artifacts/neural_candidate_ranker")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--activation", type=str, default="relu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_ranker_examples(args.ranking_data)
    if not examples:
        print(f"No ranking examples found at {args.ranking_data}")
        print("Generate ranking data via the self-improvement loop first.")
        # Save empty ranker for consistency
        ranker = NeuralCandidateRanker(
            input_dim=len(DEFAULT_RANKER_FEATURES),
            hidden_dim=args.hidden_dim,
            activation=args.activation,
        )
        config = {
            "input_dim": len(DEFAULT_RANKER_FEATURES),
            "hidden_dim": args.hidden_dim,
            "activation": args.activation,
            "features": DEFAULT_RANKER_FEATURES,
        }
        save_ranker(ranker, output_dir, config, {"status": "no_data", "examples": 0})
        print(json.dumps({"status": "no_data"}, indent=2))
        return

    dataset = build_ranker_dataset(examples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ranker = NeuralCandidateRanker(
        input_dim=len(DEFAULT_RANKER_FEATURES),
        hidden_dim=args.hidden_dim,
        activation=args.activation,
    )
    optimizer = torch.optim.Adam(ranker.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss()

    start = time.time()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        ranker.train()
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            optimizer.zero_grad()
            scores = ranker(batch["features"])
            loss = loss_fn(scores, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["features"])
            total_items += len(batch["features"])
        avg_loss = total_loss / max(total_items, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in ranker.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:02d}/{args.epochs} — Loss: {avg_loss:.4f}")

    elapsed = time.time() - start

    if best_state:
        ranker.load_state_dict(best_state)

    config = {
        "input_dim": len(DEFAULT_RANKER_FEATURES),
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "features": DEFAULT_RANKER_FEATURES,
    }
    report = {
        "status": "trained",
        "examples": len(examples),
        "epochs": args.epochs,
        "best_loss": best_loss,
        "training_time_seconds": elapsed,
    }
    save_ranker(ranker, output_dir, config, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
