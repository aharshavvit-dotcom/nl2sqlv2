from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch, load_jsonl
from neural_ir.ir_label_encoder import IRLabelEncoder, INTENTS
from neural_ir.model import DEFAULT_CONFIG, OptionAIRModel
from neural_ir.model_registry import save_model_bundle
from neural_ir.schema_linearizer import SchemaLinearizer, schema_from_example
from neural_ir.tokenizer import tokenize
from neural_ir.trainer import OptionAIRTrainer
from neural_ir.vocab import Vocabulary


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "option_a_ir_model"


def train_option_a_model(
    train_path: Path,
    validation_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_examples: int | None = None,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 13,
) -> dict[str, Any]:
    train_rows = _supported_rows(load_jsonl(train_path), max_examples=max_examples)
    validation_rows = _supported_rows(load_jsonl(validation_path), max_examples=max_examples)
    if not train_rows:
        raise ValueError(f"No supported Option A training rows found in {train_path}")
    if not validation_rows:
        validation_rows = train_rows[: min(len(train_rows), max(batch_size, 1))]

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_train = output_dir / "_train_rows.jsonl"
    temp_validation = output_dir / "_validation_rows.jsonl"
    _write_jsonl(temp_train, train_rows)
    _write_jsonl(temp_validation, validation_rows)

    config = {
        **DEFAULT_CONFIG,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vocab = Vocabulary(min_freq=1)
    vocab.build(_token_sequences(train_rows))
    label_encoder = IRLabelEncoder()
    label_encoder.fit(train_rows)

    train_dataset = IRTrainingDataset(
        str(temp_train),
        vocab=vocab,
        label_encoder=label_encoder,
        max_question_len=int(config["max_question_len"]),
        max_schema_len=int(config["max_schema_len"]),
        max_tables=int(config["max_tables"]),
        max_columns=int(config["max_columns"]),
    )
    validation_dataset = IRTrainingDataset(
        str(temp_validation),
        vocab=vocab,
        label_encoder=label_encoder,
        max_question_len=int(config["max_question_len"]),
        max_schema_len=int(config["max_schema_len"]),
        max_tables=int(config["max_tables"]),
        max_columns=int(config["max_columns"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ir_batch, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ir_batch)
    model = OptionAIRModel(config=config, vocab_size=len(vocab), label_sizes=label_encoder.label_sizes)
    trainer = OptionAIRTrainer(model, config)
    metrics = trainer.train(train_loader, validation_loader, label_encoder, output_dir)
    save_model_bundle(model, vocab, label_encoder, config, output_dir)
    metrics.update(
        {
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "vocab_size": len(vocab),
            "output_dir": str(output_dir),
        }
    )
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for temp_path in [temp_train, temp_validation]:
        if temp_path.exists():
            temp_path.unlink()
    return metrics


def _supported_rows(rows: list[dict[str, Any]], max_examples: int | None) -> list[dict[str, Any]]:
    supported = []
    for row in rows:
        query_ir = row.get("query_ir") or {}
        intent = query_ir.get("template_id") or query_ir.get("intent") or row.get("template_id") or row.get("intent")
        if intent in INTENTS:
            supported.append(row)
        if max_examples is not None and len(supported) >= max_examples:
            break
    return supported


def _token_sequences(rows: list[dict[str, Any]]) -> list[list[str]]:
    linearizer = SchemaLinearizer()
    sequences = []
    for row in rows:
        schema = schema_from_example(row)
        schema_text = row.get("serialized_schema") or linearizer.linearize(schema)
        sequences.append(tokenize(row.get("question", "")))
        sequences.append(tokenize(schema_text))
    return sequences


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight Option A QueryIR model.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "processed" / "ir_training_examples.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "processed" / "ir_validation_examples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_option_a_model(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
