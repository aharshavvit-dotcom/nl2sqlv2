from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from dataset_training.reporting import save_report_pair
from dataset_training.utils import read_jsonl, write_jsonl

from .reporting import CapabilityDatasetReporter
from .sql_capability_extractor import SQLCapabilityExtractor


DEFAULT_SPLITS = ("train", "validation", "model_selection_validation", "test", "unseen_db_test", "unsupported")


def build_capability_artifacts(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    artifact_dir: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
    dialect: str = "sqlite",
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    artifact_path = Path(artifact_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    extractor = SQLCapabilityExtractor(dialect=dialect)
    annotated_rows: list[dict[str, Any]] = []
    for split in splits:
        rows = read_jsonl(input_path / f"generic_ir_{split}.jsonl")
        for row in rows:
            sql = row.get("source_sql") or row.get("gold_sql") or row.get("sql") or ""
            annotation = extractor.extract(
                str(sql),
                example_id=str(row.get("example_id") or row.get("source_example_id") or "unknown"),
                dataset_source=str(row.get("dataset_name") or row.get("dataset") or row.get("source_dataset") or "unknown"),
                database_identifier=str(row.get("db_id") or row.get("database_id") or "unknown"),
                schema=row.get("schema"),
                sql_dialect=str(row.get("dialect") or dialect),
                full_query_ir_supported=bool(row.get("query_ir") and not row.get("unsupported_reason")),
                unsupported_reason=row.get("unsupported_reason"),
            )
            payload = annotation.model_dump(mode="json")
            annotated_rows.append(
                {
                    "example_id": row.get("example_id"),
                    "dataset_name": row.get("dataset_name") or row.get("dataset"),
                    "db_id": row.get("db_id") or row.get("database_id"),
                    "split": split,
                    "source_split": row.get("source_split"),
                    "source_sql": sql,
                    "unsupported_reason": row.get("unsupported_reason"),
                    "required_capabilities": payload["required_capabilities"],
                    "supported_capabilities": payload["supported_capabilities"],
                    "currently_supported": payload["currently_supported"],
                    "unsupported_required_capabilities": payload["unsupported_required_capabilities"],
                    "safety_labels": payload["safety_labels"],
                    "partial_supervision": payload["partial_supervision"],
                    "task_masks": payload["task_masks"],
                    "capability_annotation": payload,
                }
            )

    annotation_file = output_path / "generic_ir_capability_annotations.jsonl"
    partial_file = output_path / "generic_ir_partial_supervision.jsonl"
    write_jsonl(annotation_file, [row["capability_annotation"] for row in annotated_rows])
    write_jsonl(
        partial_file,
        [
            {
                "example_id": row["example_id"],
                "dataset_name": row["dataset_name"],
                "db_id": row["db_id"],
                "split": row["split"],
                "partial_supervision": row["partial_supervision"],
                "task_masks": row["task_masks"],
            }
            for row in annotated_rows
        ],
    )

    reporter = CapabilityDatasetReporter()
    capability_report = reporter.build_report(annotated_rows)
    retention_report = reporter.build_retention_report(annotated_rows)
    save_report_pair(artifact_path / "capability_distribution_report.json", capability_report, "Capability Distribution Report")
    save_report_pair(artifact_path / "unsupported_example_retention_report.json", retention_report, "Unsupported Example Retention Report")

    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "artifact_dir": str(artifact_path),
        "annotation_file": str(annotation_file),
        "partial_supervision_file": str(partial_file),
        "total_examples": len(annotated_rows),
        "capability_distribution_report": capability_report,
        "unsupported_example_retention_report": retention_report,
    }


def dump_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=True)
