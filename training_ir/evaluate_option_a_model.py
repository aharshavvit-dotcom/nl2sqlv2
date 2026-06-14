from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from neural_ir.evaluator import OptionAIREvaluator
from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch
from neural_ir.model_registry import load_model_bundle


DEFAULT_MODEL_DIR = ROOT / "artifacts" / "option_a_ir_model"


def evaluate_option_a_model(model_dir: Path, test_path: Path, output_path: Path) -> dict:
    bundle = load_model_bundle(model_dir)
    config = bundle["config"]
    dataset = IRTrainingDataset(
        str(test_path),
        vocab=bundle["vocab"],
        label_encoder=bundle["label_encoder"],
        max_question_len=int(config.get("max_question_len", 64)),
        max_schema_len=int(config.get("max_schema_len", 256)),
        max_tables=int(config.get("max_tables", 64)),
        max_columns=int(config.get("max_columns", 256)),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 16)), shuffle=False, collate_fn=collate_ir_batch)
    report = OptionAIREvaluator().evaluate(bundle["model"], loader, bundle["label_encoder"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Option A QueryIR model.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "ir_test_examples.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR / "evaluation_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_option_a_model(args.model_dir, args.test, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
