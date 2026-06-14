from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_loader import DatasetLoader
from datasets.models import DatabaseSchema, Text2SQLExample
from ir.sql_to_ir_converter import SQLToIRConverter
from scripts.dataset_paths import PROCESSED_DATA_DIR, parse_dataset_list


DEFAULT_OUTPUT_DIR = PROCESSED_DATA_DIR
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "option_a_ir_data"
DEFAULT_SPLIT_RATIO = (0.8, 0.1, 0.1)


class IRTrainingDataBuilder:
    def __init__(
        self,
        loader: DatasetLoader | None = None,
        converter: SQLToIRConverter | None = None,
    ):
        self.loader = loader or DatasetLoader()
        self.converter = converter or SQLToIRConverter()

    def build(
        self,
        dataset_names: list[str],
        max_examples: int | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
        include_unsupported: bool = True,
        split_ratio: tuple[float, float, float] = DEFAULT_SPLIT_RATIO,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        examples, schemas = self._load_examples(dataset_names, max_examples=max_examples)
        supported_rows: list[dict[str, Any]] = []
        unsupported_rows: list[dict[str, Any]] = []

        for example in examples:
            schema = schemas.get(example.db_id) or example.schema
            result = self.converter.convert(
                question=example.question,
                sql=example.sql,
                schema=schema,
                dataset_name=example.dataset_name,
                db_id=example.db_id,
                example_id=example.example_id,
                split=example.split,
            )
            if result["success"]:
                supported_rows.append(self._supported_row(example, schema, result))
            else:
                unsupported_rows.append(self._unsupported_row(example, result))

        split_rows = self._split_supported_rows(supported_rows, split_ratio)
        self._write_jsonl(output_dir / "ir_training_examples.jsonl", split_rows["train"])
        self._write_jsonl(output_dir / "ir_validation_examples.jsonl", split_rows["validation"])
        self._write_jsonl(output_dir / "ir_test_examples.jsonl", split_rows["test"])
        self._write_jsonl(output_dir / "ir_unsupported_examples.jsonl", unsupported_rows if include_unsupported else [])

        stats = self._stats(supported_rows, unsupported_rows, split_rows)
        self._write_json(output_dir / "ir_dataset_stats.json", stats)
        report = {
            **stats,
            "output_files": {
                "training": str(output_dir / "ir_training_examples.jsonl"),
                "validation": str(output_dir / "ir_validation_examples.jsonl"),
                "test": str(output_dir / "ir_test_examples.jsonl"),
                "unsupported": str(output_dir / "ir_unsupported_examples.jsonl"),
                "stats": str(output_dir / "ir_dataset_stats.json"),
            },
        }
        self._write_json(artifact_dir / "ir_corpus_report.json", report)
        return report

    def _load_examples(
        self,
        dataset_names: list[str],
        max_examples: int | None,
    ) -> tuple[list[Text2SQLExample], dict[str, DatabaseSchema]]:
        requested = parse_dataset_list(dataset_names)
        all_examples: list[Text2SQLExample] = []
        schemas: dict[str, DatabaseSchema] = {}
        for dataset_name in requested:
            examples, dataset_schemas = self.loader.load(dataset_name, max_examples=max_examples)
            if max_examples is not None:
                examples = examples[:max_examples]
            all_examples.extend(examples)
            schemas.update(dataset_schemas)
        return all_examples, schemas

    @staticmethod
    def _supported_row(example: Text2SQLExample, schema: Any, result: dict[str, Any]) -> dict[str, Any]:
        query_ir = result["query_ir"]
        return {
            "example_id": example.example_id,
            "dataset_name": example.dataset_name,
            "db_id": example.db_id,
            "split": example.split,
            "question": example.question,
            "serialized_schema": serialized_schema(schema),
            "source_sql": example.sql,
            "query_ir": query_ir,
            "rendered_sql": result["roundtrip_sql"],
            "intent": query_ir.get("intent"),
            "template_id": query_ir.get("template_id"),
            "roundtrip_validation": result.get("roundtrip_validation"),
            "metadata": {
                "difficulty": example.difficulty,
                "source_file": example.source_file,
                "source_split": example.split,
                "conversion_warnings": result.get("warnings", []),
            },
        }

    @staticmethod
    def _unsupported_row(example: Text2SQLExample, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "example_id": example.example_id,
            "dataset_name": example.dataset_name,
            "db_id": example.db_id,
            "question": example.question,
            "source_sql": example.sql,
            "unsupported_reason": result.get("unsupported_reason"),
            "error_message": result.get("error_message"),
            "metadata": {
                "difficulty": example.difficulty,
                "source_file": example.source_file,
                "split": example.split,
                "warnings": result.get("warnings", []),
            },
        }

    @staticmethod
    def _split_supported_rows(
        rows: list[dict[str, Any]],
        split_ratio: tuple[float, float, float],
    ) -> dict[str, list[dict[str, Any]]]:
        total = len(rows)
        train_count = int(total * split_ratio[0])
        validation_count = int(total * split_ratio[1])
        splits = {
            "train": [dict(row, split="train") for row in rows[:train_count]],
            "validation": [dict(row, split="validation") for row in rows[train_count : train_count + validation_count]],
            "test": [dict(row, split="test") for row in rows[train_count + validation_count :]],
        }
        for split, split_rows in splits.items():
            for row in split_rows:
                row["query_ir"]["metadata"]["split"] = split
        return splits

    @staticmethod
    def _stats(
        supported_rows: list[dict[str, Any]],
        unsupported_rows: list[dict[str, Any]],
        split_rows: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        all_rows = [*supported_rows, *unsupported_rows]
        return {
            "total_examples": len(all_rows),
            "successful_examples": len(supported_rows),
            "unsupported_examples": len(unsupported_rows),
            "conversion_success_rate": len(supported_rows) / len(all_rows) if all_rows else 0.0,
            "by_dataset": dict(Counter(row["dataset_name"] for row in supported_rows)),
            "by_intent": dict(Counter(row["intent"] for row in supported_rows)),
            "by_split": {split: len(rows) for split, rows in split_rows.items()},
            "by_unsupported_reason": dict(Counter(row.get("unsupported_reason") for row in unsupported_rows)),
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serialized_schema(schema: Any) -> str | None:
    if schema is None:
        return None
    if isinstance(schema, DatabaseSchema):
        return schema.serialized_schema
    if isinstance(schema, dict):
        return schema.get("serialized_schema")
    return getattr(schema, "serialized_schema", None)


def parse_split_ratio(value: str | None) -> tuple[float, float, float]:
    if not value:
        return DEFAULT_SPLIT_RATIO
    parts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            item = item.split("=", 1)[1]
        parts.append(float(item))
    if len(parts) != 3 or sum(parts) <= 0:
        raise ValueError("--split-ratio must contain three positive values")
    total = sum(parts)
    return parts[0] / total, parts[1] / total, parts[2] / total


def build_ir_training_data(
    datasets: list[str],
    max_examples: int | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    include_unsupported: bool = True,
    split_ratio: tuple[float, float, float] = DEFAULT_SPLIT_RATIO,
    loader: DatasetLoader | None = None,
) -> dict[str, Any]:
    return IRTrainingDataBuilder(loader=loader).build(
        dataset_names=datasets,
        max_examples=max_examples,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        include_unsupported=include_unsupported,
        split_ratio=split_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build QueryIR labels from public Text-to-SQL datasets.")
    parser.add_argument("--datasets", default="wikisql,spider,bird-mini")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--include-unsupported", action="store_true", default=True)
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_ir_training_data(
        datasets=parse_dataset_list(args.datasets),
        max_examples=args.max_examples,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        include_unsupported=args.include_unsupported,
        split_ratio=parse_split_ratio(args.split_ratio),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
