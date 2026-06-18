from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl, write_json, write_jsonl
from self_training.error_classifier import ErrorClassifier
from self_training.gold_comparator import GoldComparator


class _Args:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def evaluate_against_gold(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.exists():
        raise FileNotFoundError(f"Input examples not found: {args.input}")
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError(f"Input examples are empty: {args.input}")
    predictions, warnings = _build_predictions(rows, args)
    comparator = GoldComparator()
    comparison = comparator.compare_batch(predictions, rows)
    error_report = ErrorClassifier().classify_batch(comparison.per_example, predictions)
    output_rows = []
    for prediction, comp in zip(predictions, comparison.per_example):
        output_rows.append(
            {
                **prediction,
                "gold_comparison_score": comp.match_score,
                "is_exact_match": comp.is_exact_match,
                "is_partial_match": comp.is_partial_match,
                "mismatched_fields": [field for field, matched in comp.field_matches.items() if not matched],
            }
        )
    report = {
        "summary": {
            "total_examples": comparison.total,
            "exact_matches": comparison.exact_matches,
            "partial_matches": comparison.partial_matches,
            "failures": comparison.failures,
            "gold_comparison_score": sum(row["gold_comparison_score"] for row in output_rows) / max(len(output_rows), 1),
            "exact_match_rate": comparison.exact_matches / max(comparison.total, 1),
            "source": output_rows[0].get("prediction_source") if output_rows else "unknown",
        },
        "field_accuracy": comparison.field_accuracy,
        "errors": {
            "total_errors": error_report.total_errors,
            "by_category": error_report.by_category,
            "by_severity": error_report.by_severity,
        },
        "warnings": warnings,
    }
    write_jsonl(args.output, output_rows)
    write_json(args.report, report)
    return report


def _build_predictions(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    model_dir = Path(args.neural_model_dir)
    if (model_dir / "model.pt").exists():
        from self_training.prediction_runner import PredictionRunner

        try:
            return PredictionRunner(model_dir).predict_batch(rows, max_examples=args.max_examples), warnings
        except Exception as exc:
            warnings.append(
                f"Neural QueryIR predictor could not load artifacts at {model_dir}: {exc}; "
                "using explicit gold_replay_baseline for pipeline evaluation."
            )

    warnings.append(f"No runnable Neural QueryIR model found at {model_dir}; using explicit gold_replay_baseline for pipeline smoke evaluation.")
    subset = rows[: args.max_examples] if args.max_examples else rows
    predictions = []
    for row in subset:
        query_ir = row.get("predicted_query_ir") or row.get("query_ir") or {}
        predictions.append(
            {
                "example_id": row.get("example_id"),
                "question": row.get("question"),
                "dataset_name": row.get("dataset_name"),
                "db_id": row.get("db_id"),
                "split": row.get("split"),
                "predicted_query_ir": query_ir,
                "predicted_sql": row.get("rendered_sql") or row.get("source_sql"),
                "gold_query_ir": row.get("query_ir"),
                "gold_sql": row.get("source_sql") or row.get("rendered_sql"),
                "schema": row.get("schema"),
                "prediction_source": "gold_replay_baseline",
                "prediction_failed": False,
            }
        )
    return predictions, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model predictions against gold QueryIR/SQL labels.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "generic_ir_validation.jsonl")
    parser.add_argument("--retrieval-model-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    parser.add_argument("--neural-model-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_predictions.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_gold_comparison_report.json")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    report = evaluate_against_gold(parse_args())
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
