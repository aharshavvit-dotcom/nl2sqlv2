from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl, write_json
from retrieval.feedback_index import FeedbackIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the feedback correction index for local RAG retrieval.")
    parser.add_argument("--feedback-examples", type=Path, default=ROOT / "data" / "processed" / "feedback_positive_examples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = read_jsonl(args.feedback_examples)
    index = FeedbackIndex()
    index.build(examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "feedback_index.pkl"
    index.save(index_path)
    report = {
        "feedback_examples": len(examples),
        "index_path": str(index_path),
        "status": "built",
    }
    write_json(args.output_dir / "feedback_index_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
