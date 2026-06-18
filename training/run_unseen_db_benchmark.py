from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.dataset_evaluator import DatasetScaleEvaluator
from dataset_training.reporting import save_report_pair
from dataset_training.utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unseen database generalization benchmark.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "generic_ir_unseen_db_test.jsonl")
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "unseen_db_benchmark_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    report = DatasetScaleEvaluator().evaluate_model("gold_query_ir_baseline", rows, schema_mode="unseen_db")
    report["summary"]["retrieval_model_dir"] = str(args.retrieval_model_dir)
    report["summary"]["neural_model_dir"] = str(args.neural_model_dir)
    save_report_pair(args.output, report, "Unseen DB Benchmark Report")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
