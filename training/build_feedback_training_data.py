from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import write_json, write_jsonl
from feedback.feedback_store import FeedbackStore
from feedback.feedback_to_ir_examples import FeedbackToIRExampleBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert user feedback into QueryIR training examples.")
    parser.add_argument("--feedback", type=Path, default=ROOT / "data" / "feedback" / "query_feedback.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "feedback")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = FeedbackStore(args.feedback).load_all()
    result = FeedbackToIRExampleBuilder().build_examples(rows)
    write_jsonl(args.output_dir / "feedback_positive_examples.jsonl", result["positive_examples"])
    write_jsonl(args.output_dir / "feedback_hard_negatives.jsonl", result["hard_negatives"])
    write_jsonl(args.output_dir / "feedback_safety_regressions.jsonl", result["safety_regressions"])
    write_json(args.artifact_dir / "feedback_training_data_report.json", result)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
