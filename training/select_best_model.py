from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_selection.model_candidate import ModelCandidate
from model_selection.model_selector import ModelSelector
from model_selection.selection_reporter import SelectionReporter
from quality_gates.model_quality_gate import ModelQualityGate
from quality_gates.thresholds import load_thresholds


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _metrics(evaluation: dict, execution: dict) -> dict:
    metrics, _present = ModelQualityGate._extract_metrics(evaluation)
    summary = execution.get("summary") or {}
    metrics.update(
        {
            "gold_comparison_score": evaluation.get("summary", {}).get("gold_comparison_score", metrics.get("query_ir_validity_rate", 0.0)),
            "execution_match_rate": summary.get("execution_match_rate", 0.0),
            "structure_match_rate": summary.get("structure_match_rate", summary.get("sql_structure_match_rate", 0.0)),
            "sql_structure_match_rate": summary.get("structure_match_rate", 0.0),
            "unnecessary_join_rate": summary.get("unnecessary_join_rate", metrics.get("unnecessary_join_rate_max", 0.0)),
            "wrong_table_rate": summary.get("wrong_table_rate", metrics.get("wrong_table_rate_max", 0.0)),
            "analytics_query_pass_rate": evaluation.get("summary", {}).get("analytics_query_pass_rate", 1.0),
            "per_example": evaluation.get("test_performance", {}).get("per_example", []),
        }
    )
    return metrics


def _attach_predicted_sql_metrics(metrics: dict, predicted: dict) -> dict:
    if not predicted:
        return metrics
    metrics.update({
        "controlled_predicted_sql_execution_match_rate": predicted.get(
            "predicted_execution_match_rate",
            predicted.get("predicted_result_value_match_rate", 0.0),
        ),
        "controlled_predicted_sql_execution_success_rate": predicted.get("predicted_execution_success_rate", 0.0),
        "controlled_predicted_sql_row_count_match_rate": predicted.get("predicted_row_count_match_rate", 0.0),
        "controlled_predicted_sql_safe_sql_rate": predicted.get("predicted_safe_sql_rate", 0.0),
        "controlled_predicted_sql_unsafe_sql_count": predicted.get(
            "unsafe_sql_count",
            predicted.get("predicted_unsafe_sql_count", 0),
        ),
    })
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best model candidate from evaluation reports.")
    parser.add_argument("--evaluation-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--execution-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "execution_aware_evaluation_report.json")
    parser.add_argument("--controlled-predicted-sql-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "controlled_predicted_sql_execution_report.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evaluation" / "model_quality_thresholds.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "model_selection_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controlled_predicted_sql_report = _read(args.controlled_predicted_sql_report)
    metrics = _attach_predicted_sql_metrics(
        _metrics(_read(args.evaluation_report), _read(args.execution_report)),
        controlled_predicted_sql_report,
    )
    # Attach multi-seed variance report if available
    variance_path = args.evaluation_report.parent / "multi_seed_variance_report.json"
    multi_seed_report = _read(variance_path) if variance_path.exists() else None
    candidate = ModelCandidate(
        name="adaptive_router",
        artifact_dir=str(ROOT / "artifacts"),
        model_type="adaptive_router",
        metrics=metrics,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        metadata={
            "evaluation_report": str(args.evaluation_report),
            "execution_report": str(args.execution_report),
            "controlled_predicted_sql_report": controlled_predicted_sql_report,
            "multi_seed_report": multi_seed_report,
        },
    )
    report = ModelSelector().select_best([candidate], load_thresholds(args.thresholds))
    SelectionReporter().write(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
