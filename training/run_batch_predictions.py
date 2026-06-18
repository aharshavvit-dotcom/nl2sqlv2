"""CLI: Run batch predictions using the neural IR model.

Usage
-----
python training/run_batch_predictions.py \\
  --model-dir artifacts/neural_ir_model \\
  --input data/processed/generic_ir_validation.jsonl \\
  --output artifacts/self_training/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_training.prediction_runner import PredictionRunner


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch predictions using the neural IR model on dataset examples.",
    )
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "artifacts" / "neural_ir_model",
                        help="Directory containing the trained neural IR model.")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "processed" / "generic_ir_validation.jsonl",
                        help="Path to input JSONL file with examples.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "self_training" / "predictions.jsonl",
                        help="Path to write prediction results.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of examples to predict (for quick iteration).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    examples = _read_jsonl(args.input)
    if not examples:
        print(f"WARNING: No examples found at {args.input}")
        return 1

    print(f"Running predictions on {len(examples)} examples...")
    print(f"Model directory: {args.model_dir}")

    runner = PredictionRunner(args.model_dir)
    predictions = runner.predict_batch(examples, max_examples=args.max_examples)

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, ensure_ascii=False, default=str) + "\n")

    # Summary stats
    successful = sum(1 for p in predictions if not p.get("prediction_failed"))
    failed = sum(1 for p in predictions if p.get("prediction_failed"))
    avg_time = sum(p.get("prediction_time_ms", 0) for p in predictions) / max(len(predictions), 1)
    avg_conf = sum(p.get("confidence", 0) for p in predictions if not p.get("prediction_failed")) / max(successful, 1)

    summary = {
        "total": len(predictions),
        "successful": successful,
        "failed": failed,
        "avg_prediction_time_ms": round(avg_time, 2),
        "avg_confidence": round(avg_conf, 4),
        "output_file": str(args.output),
    }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
