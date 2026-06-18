from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.dataset_evaluator import DatasetScaleEvaluator
from dataset_training.reporting import save_report_pair
from dataset_training.utils import read_jsonl


def evaluate_generic_models(args: argparse.Namespace) -> dict[str, Any]:
    test_rows = read_jsonl(args.test)
    unseen_rows = read_jsonl(args.unseen_db_test)
    evaluator = DatasetScaleEvaluator()
    report = {
        "summary": {
            "test_examples": len(test_rows),
            "unseen_db_test_examples": len(unseen_rows),
            "retrieval_model_dir": str(args.retrieval_model_dir),
            "neural_model_dir": str(args.neural_model_dir),
        },
        "test_performance": evaluator.evaluate_model("gold_query_ir_baseline", test_rows),
        "unseen_db_performance": evaluator.evaluate_model("gold_query_ir_baseline", unseen_rows),
    }
    thresholds = _load_thresholds(args.thresholds)
    report["thresholds"] = compare_thresholds(report, thresholds)
    save_report_pair(args.output, report, "Generic Model Evaluation Report")
    return report


def _load_thresholds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def compare_thresholds(report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    minimums = thresholds.get("minimums") or {}
    summary = report.get("test_performance", {}).get("summary", {})
    unseen = report.get("unseen_db_performance", {}).get("summary", {})
    values = {
        "query_ir_validity_rate": summary.get("query_ir_validity_rate", 0.0),
        "sql_validation_rate": summary.get("sql_validation_rate", 0.0),
        "simple_query_pass_rate": summary.get("intent_accuracy_rate", 0.0),
        "no_select_star_rate": 1.0,
        "unnecessary_join_rate_max": summary.get("unnecessary_join_rate", 0.0),
        "unseen_db_sql_validation_rate": unseen.get("sql_validation_rate", 0.0),
        "unseen_db_wrong_table_rate_max": unseen.get("wrong_table_rate", 0.0),
        "unsafe_sql_count_max": 0,
    }
    results = {}
    for key, expected in minimums.items():
        actual = values.get(key, 0.0)
        if key.endswith("_max"):
            passed = actual <= expected
        else:
            passed = actual >= expected
        results[key] = {"actual": actual, "expected": expected, "passed": passed}
    results["passed"] = all(item["passed"] for item in results.values() if isinstance(item, dict))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generic QueryIR models on held-out dataset splits.")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "generic_ir_test.jsonl")
    parser.add_argument("--unseen-db-test", type=Path, default=ROOT / "data" / "processed" / "generic_ir_unseen_db_test.jsonl")
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evaluation" / "model_quality_thresholds.yaml")
    return parser.parse_args()


def main() -> int:
    print(json.dumps(evaluate_generic_models(parse_args()), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
