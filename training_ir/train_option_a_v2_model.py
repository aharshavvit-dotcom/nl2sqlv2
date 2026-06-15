from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.attention_model import DEFAULT_V2_CONFIG, SchemaAwareOptionAIRModel
from neural_ir.candidate_builder import SchemaCandidateBuilder
from neural_ir.confidence_calibrator import OptionAConfidenceCalibrator
from neural_ir.evaluator import OptionAIREvaluator
from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch, load_jsonl
from neural_ir.ir_label_encoder import IRLabelEncoder, INTENTS
from neural_ir.model_registry import save_model_bundle
from neural_ir.schema_linearizer import SchemaLinearizer, schema_from_example
from neural_ir.tokenizer import tokenize
from neural_ir.trainer import OptionAIRTrainer
from neural_ir.vocab import Vocabulary


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "option_a_ir_model_v2"


def train_option_a_v2_model(
    train_path: Path,
    validation_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hard_negatives_path: Path | None = None,
    max_examples: int | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.0007,
    seed: int = 13,
    use_hard_negative_loss: bool = True,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_rows = _supported_rows(load_jsonl(train_path), max_examples=max_examples)
    validation_rows = _supported_rows(load_jsonl(validation_path), max_examples=max_examples)
    if not train_rows:
        raise ValueError(f"No supported Option A V2 training rows found in {train_path}")
    if not validation_rows:
        validation_rows = train_rows[: min(len(train_rows), max(batch_size, 1))]
    hard_negative_rows = load_jsonl(hard_negatives_path) if hard_negatives_path else []

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_train = output_dir / "_train_rows.jsonl"
    temp_validation = output_dir / "_validation_rows.jsonl"
    _write_jsonl(temp_train, train_rows)
    _write_jsonl(temp_validation, validation_rows)

    config = {
        **DEFAULT_V2_CONFIG,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "use_hard_negative_loss": use_hard_negative_loss,
        **(model_overrides or {}),
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vocab = Vocabulary(min_freq=1)
    vocab.build(_token_sequences(train_rows + validation_rows))
    label_encoder = IRLabelEncoder()
    label_encoder.fit(train_rows)

    dataset_kwargs = {
        "vocab": vocab,
        "label_encoder": label_encoder,
        "max_question_len": int(config["max_question_len"]),
        "max_schema_len": int(config["max_schema_len"]),
        "max_candidate_tokens": int(config["max_candidate_tokens"]),
        "max_tables": int(config["max_tables"]),
        "max_columns": int(config["max_columns"]),
    }
    train_dataset = IRTrainingDataset(str(temp_train), **dataset_kwargs)
    validation_dataset = IRTrainingDataset(str(temp_validation), **dataset_kwargs)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ir_batch, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ir_batch)

    model = SchemaAwareOptionAIRModel(config=config, vocab_size=len(vocab), label_sizes=label_encoder.label_sizes)
    trainer = OptionAIRTrainer(model, config)
    metrics = trainer.train(train_loader, validation_loader, output_dir, label_encoder)
    save_model_bundle(model, vocab, label_encoder, config, output_dir)

    evaluation_report = _evaluate_safely(model, validation_loader, label_encoder)
    (output_dir / "evaluation_report.json").write_text(json.dumps(evaluation_report, indent=2, ensure_ascii=False), encoding="utf-8")
    calibrator = OptionAConfidenceCalibrator()
    calibrator.fit([])
    calibrator.save(str(output_dir / "option_a_calibration.json"))

    metrics.update(
        {
            "model_version": "option_a_v2",
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "hard_negative_examples": len(hard_negative_rows),
            "vocab_size": len(vocab),
            "output_dir": str(output_dir),
        }
    )
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    for temp_path in [temp_train, temp_validation]:
        if temp_path.exists():
            temp_path.unlink()
    return metrics


def _evaluate_safely(model, validation_loader, label_encoder) -> dict[str, Any]:
    try:
        report = OptionAIREvaluator().evaluate(model, validation_loader, label_encoder)
        return {
            "summary": report,
            "by_intent": report.get("by_intent", {}),
            "sample_failures": report.get("sample_failures", []),
            "recommendations": [],
        }
    except Exception as exc:
        return {"summary": {"error": str(exc)}, "by_intent": {}, "sample_failures": [], "recommendations": ["inspect validation conversion failures"]}


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
    candidate_builder = SchemaCandidateBuilder()
    sequences = []
    for row in rows:
        schema = schema_from_example(row)
        schema_text = row.get("serialized_schema") or linearizer.linearize(schema)
        sequences.append(tokenize(row.get("question", "")))
        sequences.append(tokenize(schema_text))
        candidates = candidate_builder.build_candidates(schema, row.get("question", ""))
        for item in [*candidates.get("tables", []), *candidates.get("columns", [])]:
            sequences.append(list(item.get("tokens") or []))
    return sequences


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Option A V2 schema-aware QueryIR model.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use-hard-negative-loss", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_option_a_v2_model(
        train_path=args.train,
        validation_path=args.validation,
        hard_negatives_path=args.hard_negatives,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        seed=args.seed,
        use_hard_negative_loss=args.use_hard_negative_loss or bool(args.hard_negatives),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
