from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_training.self_improvement_loop import SelfImprovementLoop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated dataset-driven self-improvement loop.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "processed" / "generic_ir_train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "processed" / "generic_ir_validation.jsonl")
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "self_training")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = SelfImprovementLoop().run(
        train_path=args.train,
        validation_path=args.validation,
        retrieval_model_dir=args.retrieval_model_dir,
        neural_model_dir=args.neural_model_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        max_examples=args.max_examples,
    )
    print(json.dumps({"iterations": report["iterations"], "improved": report["improved"]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
