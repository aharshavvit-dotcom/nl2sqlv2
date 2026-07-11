from __future__ import annotations

from capabilities import CapabilityDatasetReporter, SQLCapabilityExtractor
from capabilities.evaluation import CapabilityEvaluator


def _row(sql: str, split: str, unsupported_reason: str | None = None) -> dict:
    annotation = SQLCapabilityExtractor().extract(
        sql,
        example_id=f"{split}:{sql[:8]}",
        dataset_source="mock",
        database_identifier="db",
        full_query_ir_supported=unsupported_reason is None,
        unsupported_reason=unsupported_reason,
    ).model_dump(mode="json")
    return {
        "dataset_name": "mock",
        "split": split,
        "unsupported_reason": unsupported_reason,
        "required_capabilities": annotation["required_capabilities"],
        "partial_supervision": annotation["partial_supervision"],
        "task_masks": annotation["task_masks"],
        "capability_annotation": annotation,
    }


def test_capability_dataset_reporting_summarizes_distribution_and_retention() -> None:
    rows = [
        _row("SELECT id FROM users", "train"),
        _row("SELECT RANK() OVER (ORDER BY amount DESC) FROM orders", "unsupported", "window_function"),
        _row("SELECT id FROM a UNION SELECT id FROM b", "unsupported", "set_operation"),
    ]
    reporter = CapabilityDatasetReporter(rare_threshold=2)

    report = reporter.build_report(rows)
    retention = reporter.build_retention_report(rows)

    assert report["summary"]["total_examples"] == 3
    assert report["summary"]["partial_supervision_only_count"] == 2
    assert report["capability_frequency"]["WINDOW_RANK"] == 1
    assert report["set_operation_distribution"]["UNION"] == 1
    assert "zero_coverage_capabilities" in report
    assert report["summary"]["partial_supervision_extraction_coverage"] == 1.0
    assert retention["summary"]["unsupported_examples"] == 2
    assert retention["summary"]["retained_for_auxiliary_supervision"] == 2


def test_capability_evaluator_computes_multilabel_metrics() -> None:
    report = CapabilityEvaluator().evaluate(
        gold_labels=[["SIMPLE_SELECT", "FILTER"], ["SIMPLE_SELECT", "AGGREGATION"]],
        predicted_scores=[
            {"SIMPLE_SELECT": 0.9, "FILTER": 0.8, "AGGREGATION": 0.1},
            {"SIMPLE_SELECT": 0.9, "FILTER": 0.2, "AGGREGATION": 0.7},
        ],
        thresholds={"SIMPLE_SELECT": 0.5, "FILTER": 0.5, "AGGREGATION": 0.5},
    )

    assert report["micro_f1"] == 1.0
    assert report["exact_multilabel_match"] == 1.0
    assert report["per_capability"]["FILTER"]["average_precision"] == 1.0
