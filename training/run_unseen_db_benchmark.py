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
from training.evaluate_generic_models import _evaluation_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unseen database generalization benchmark.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "generic_ir_unseen_db_test.jsonl")
    parser.add_argument("--model-bundle-dir", type=Path, default=None)
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--allow-gold-replay-baseline", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "unseen_db_benchmark_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    evaluation_rows, source, artifact_source = _evaluation_rows(
        rows,
        args.retrieval_model_dir,
        args.neural_model_dir,
        model_bundle_dir=args.model_bundle_dir,
        allow_gold_replay_baseline=args.allow_gold_replay_baseline,
    )
    mode = "explicit_gold_replay_baseline" if source == "gold_replay_baseline" else "real_model_predictions"
    report = DatasetScaleEvaluator().evaluate_model(
        source,
        evaluation_rows,
        schema_mode="unseen_db",
        evaluation_mode=mode,
        model_artifact_source=artifact_source,
        predictor_used=mode == "real_model_predictions",
    )
    report["model_bundle_dir"] = str(args.model_bundle_dir or "")
    report["retrieval_model_dir"] = str(args.retrieval_model_dir)
    report["neural_model_dir"] = str(args.neural_model_dir)
    report["summary"]["model_bundle_dir"] = report["model_bundle_dir"]
    report["summary"]["retrieval_model_dir"] = report["retrieval_model_dir"]
    report["summary"]["neural_model_dir"] = report["neural_model_dir"]
    report["summary"]["test_source"] = report["test_source"]
    report["summary"]["real_predictions_generated"] = report["real_predictions_generated"]
    report["summary"]["prediction_failures"] = report["prediction_failures"]
    save_report_pair(args.output, report, "Unseen DB Benchmark Report")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
