from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from datasets.dataset_loader import DatasetLoader
from datasets.models import DatabaseSchema, Text2SQLExample
from datasets.sql_feature_extractor import SQLFeatureExtractor
from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.sql_to_ir_converter import SQLToIRConverter
from validation.sql_validator import SQLValidator

from .corpus_quality import CorpusQualityAnalyzer
from .dataset_registry import DatasetRegistry
from .leakage_checker import DatasetLeakageChecker
from .reporting import save_report_pair
from .split_manager import DatasetSplitManager
from .utils import model_dump, normalize_dataset_name, write_jsonl


class GenericIRCorpusBuilder:
    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        split_manager: DatasetSplitManager,
        sql_to_ir_converter: SQLToIRConverter | None = None,
        quality_filter: Any | None = None,
    ):
        self.dataset_registry = dataset_registry
        self.split_manager = split_manager
        self.sql_to_ir_converter = sql_to_ir_converter or SQLToIRConverter()
        self.quality_filter = quality_filter
        self.extractor = SQLFeatureExtractor()
        self.renderer = IRToSQLRenderer()
        self.sql_validator = SQLValidator()

    def build(
        self,
        datasets: list[str],
        max_examples: int | None,
        output_dir: str,
        artifact_dir: str,
        max_examples_per_dataset: dict[str, int] | None = None,
        min_converted_examples_required: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        output = Path(output_dir)
        artifacts = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        requested = [normalize_dataset_name(item) for item in datasets]
        registry_report = self.dataset_registry.validate_dataset_presence(requested)
        examples, schemas = self._load_examples(
            requested,
            registry_report,
            max_examples=max_examples,
            max_examples_per_dataset=max_examples_per_dataset,
        )

        supported_rows: list[dict[str, Any]] = []
        unsupported_rows: list[dict[str, Any]] = []
        for example in examples:
            schema = schemas.get(example.db_id) or example.schema
            result = self.sql_to_ir_converter.convert(
                question=example.question,
                sql=example.sql,
                schema=schema,
                dataset_name=example.dataset_name,
                db_id=example.db_id,
                example_id=example.example_id,
                split=example.split,
            )
            if result.get("success"):
                row = self._supported_row(example, schema, result)
                if self.quality_filter and not self.quality_filter(row):
                    unsupported_rows.append(self._unsupported_row(example, result, reason="quality_filter_rejected"))
                else:
                    supported_rows.append(row)
            else:
                unsupported_rows.append(self._unsupported_row(example, result))

        splits = self.split_manager.split_by_database([*supported_rows, *unsupported_rows])
        self._write_split_files(output, splits)

        leakage_report = DatasetLeakageChecker().run_all_checks(splits)
        quality_report = CorpusQualityAnalyzer().analyze(
            [row for name in ["train", "validation", "test", "unseen_db_test"] for row in splits[name]],
            splits["unsupported"],
        )
        split_report = {
            "datasets_requested": requested,
            "dataset_registry": registry_report,
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "databases": {name: sorted({str(row.get("db_id")) for row in rows if row.get("db_id")}) for name, rows in splits.items()},
        }

        save_report_pair(artifacts / "dataset_split_report.json", split_report, "Dataset Split Report")
        save_report_pair(artifacts / "leakage_report.json", leakage_report, "Dataset Leakage Report")
        save_report_pair(artifacts / "corpus_quality_report.json", quality_report, "Corpus Quality Report")
        contribution_report = self._dataset_contribution_report(
            requested=requested,
            registry_report=registry_report,
            examples=examples,
            splits=splits,
            leakage_report=leakage_report,
            min_converted_examples_required=min_converted_examples_required,
        )
        unsupported_report = self._unsupported_sql_report(unsupported_rows)
        save_report_pair(artifacts / "dataset_contribution_report.json", contribution_report, "Dataset Contribution Report")
        save_report_pair(artifacts / "unsupported_sql_report.json", unsupported_report, "Unsupported SQL Report")

        return {
            "output_dir": str(output),
            "artifact_dir": str(artifacts),
            "split_report": split_report,
            "leakage_report": leakage_report,
            "corpus_quality_report": quality_report,
            "dataset_contribution_report": contribution_report,
            "unsupported_sql_report": unsupported_report,
            "output_files": {
                name: str(output / f"generic_ir_{name}.jsonl")
                for name in ["train", "validation", "test", "unseen_db_test", "unsupported"]
            },
        }

    def _load_examples(
        self,
        datasets: list[str],
        registry_report: dict[str, dict[str, Any]],
        max_examples: int | None,
        max_examples_per_dataset: dict[str, int] | None = None,
    ) -> tuple[list[Text2SQLExample], dict[str, DatabaseSchema]]:
        normalized_limits = {
            normalize_dataset_name(name): int(limit)
            for name, limit in (max_examples_per_dataset or {}).items()
            if limit is not None and int(limit) > 0
        }
        loader = DatasetLoader(raw_root=self.dataset_registry.root_dir)
        all_examples: list[Text2SQLExample] = []
        schemas: dict[str, DatabaseSchema] = {}
        for dataset_name in datasets:
            if not registry_report.get(dataset_name, {}).get("available"):
                continue
            dataset_limit = normalized_limits.get(dataset_name, max_examples)
            try:
                if hasattr(self.dataset_registry, "load_examples"):
                    examples, dataset_schemas = self.dataset_registry.load_examples(  # type: ignore[attr-defined]
                        [dataset_name],
                        max_examples=dataset_limit,
                    )
                else:
                    examples, dataset_schemas = loader.load(dataset_name, max_examples=dataset_limit)
            except Exception:
                continue
            if dataset_limit is not None:
                examples = examples[:dataset_limit]
            all_examples.extend(examples)
            schemas.update(dataset_schemas)
        return all_examples, schemas

    def _supported_row(self, example: Text2SQLExample, schema: Any, result: dict[str, Any]) -> dict[str, Any]:
        query_ir = result["query_ir"]
        rendered_sql = result.get("roundtrip_sql") or self.renderer.render(query_ir)
        sql_features = self.extractor.extract(example.sql)
        return {
            "example_id": example.example_id,
            "dataset_name": example.dataset_name,
            "db_id": example.db_id,
            "split": example.split,
            "question": example.question,
            "serialized_schema": self._serialized_schema(schema),
            "schema": self._schema_dict(schema),
            "source_sql": example.sql,
            "query_ir": query_ir,
            "rendered_sql": rendered_sql,
            "intent": query_ir.get("intent"),
            "template_id": query_ir.get("template_id"),
            "complexity": sql_features.get("complexity", "unknown"),
            "sql_features": sql_features,
            "ir_validation": result.get("ir_validation"),
            "sql_validation": result.get("sql_validation"),
            "roundtrip_validation": result.get("roundtrip_validation"),
            "metadata": {
                "difficulty": example.difficulty,
                "source_file": example.source_file,
                "source_split": example.split,
                "conversion_warnings": result.get("warnings", []),
            },
        }

    @staticmethod
    def _unsupported_row(
        example: Text2SQLExample,
        result: dict[str, Any],
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "example_id": example.example_id,
            "dataset": example.dataset_name,
            "dataset_name": example.dataset_name,
            "db_id": example.db_id,
            "question": example.question,
            "gold_sql": example.sql,
            "source_sql": example.sql,
            "unsupported_reason": reason or result.get("unsupported_reason") or "unsupported",
            "unsupported_feature": GenericIRCorpusBuilder._unsupported_feature(
                reason or result.get("unsupported_reason") or "unsupported",
                result.get("error_message"),
            ),
            "error_message": result.get("error_message"),
            "metadata": {
                "difficulty": example.difficulty,
                "source_file": example.source_file,
                "split": example.split,
                "warnings": result.get("warnings", []),
            },
        }

    @staticmethod
    def _dataset_contribution_report(
        requested: list[str],
        registry_report: dict[str, dict[str, Any]],
        examples: list[Text2SQLExample],
        splits: dict[str, list[dict[str, Any]]],
        leakage_report: dict[str, Any],
        min_converted_examples_required: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        by_dataset: dict[str, dict[str, Any]] = {}
        all_names = list(dict.fromkeys([*requested, "wikisql", "spider", "bird-mini", "bird-full"]))
        raw_counts = Counter(example.dataset_name for example in examples)
        loaded_counts = Counter(example.dataset_name for example in examples)
        minimums = {
            normalize_dataset_name(name): int(value)
            for name, value in (min_converted_examples_required or {}).items()
            if value is not None and int(value) > 0
        }
        minimum_failures: list[dict[str, Any]] = []
        for name in all_names:
            split_counts = {
                split_name: sum(1 for row in splits.get(split_name, []) if row.get("dataset_name") == name)
                for split_name in ["train", "validation", "test", "unseen_db_test", "unsupported"]
            }
            converted = int(
                split_counts["train"]
                + split_counts["validation"]
                + split_counts["test"]
                + split_counts["unseen_db_test"]
            )
            minimum_required = int(minimums.get(name, 0))
            if minimum_required and converted < minimum_required:
                minimum_failures.append(
                    {
                        "dataset": name,
                        "converted_to_queryir": converted,
                        "minimum_required": minimum_required,
                    }
                )
            unsupported_reasons = Counter(
                row.get("unsupported_reason") or "unsupported"
                for row in splits.get("unsupported", [])
                if row.get("dataset_name") == name
            )
            by_dataset[name] = {
                "raw_examples": int(raw_counts.get(name, 0)),
                "loaded_examples": int(loaded_counts.get(name, 0)),
                "converted_to_queryir": converted,
                "used_in_train": int(split_counts["train"]),
                "used_in_validation": int(split_counts["validation"]),
                "used_in_test": int(split_counts["test"]),
                "used_in_unseen_db_test": int(split_counts["unseen_db_test"]),
                "unsupported": int(split_counts["unsupported"]),
                "unsupported_reasons": dict(unsupported_reasons),
                "minimum_required": minimum_required,
                "minimum_passed": converted >= minimum_required,
            }
        return {
            "datasets_requested": requested,
            "datasets_found": [name for name in requested if registry_report.get(name, {}).get("available")],
            "datasets_missing": [name for name in requested if not registry_report.get(name, {}).get("available")],
            "by_dataset": by_dataset,
            "total_training_examples": len(splits.get("train", [])),
            "leakage_check_passed": bool(
                leakage_report.get("passed", leakage_report.get("ok", not leakage_report.get("has_leakage", False)))
            ),
            "minimums": minimums,
            "minimum_failures": minimum_failures,
            "full_training_dataset_minimums_passed": not minimum_failures,
        }

    @staticmethod
    def _unsupported_sql_report(unsupported_rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_dataset = Counter(row.get("dataset_name") or row.get("dataset") or "unknown" for row in unsupported_rows)
        by_feature = Counter(row.get("unsupported_feature") or row.get("unsupported_reason") or "unsupported" for row in unsupported_rows)
        total = len(unsupported_rows)
        return {
            "summary": {
                "unsupported_examples": total,
                "datasets": len(by_dataset),
                "features": len(by_feature),
            },
            "unsupported_by_dataset": dict(by_dataset),
            "unsupported_by_feature": dict(by_feature),
            "training_data_loss_by_feature": {
                feature: {
                    "count": count,
                    "share_of_unsupported": count / total if total else 0.0,
                }
                for feature, count in by_feature.items()
            },
            "top_20_examples": [
                {
                    "dataset": row.get("dataset_name") or row.get("dataset"),
                    "db_id": row.get("db_id"),
                    "question": row.get("question"),
                    "gold_sql": row.get("gold_sql") or row.get("source_sql"),
                    "unsupported_reason": row.get("unsupported_reason"),
                    "unsupported_feature": row.get("unsupported_feature"),
                    "error_message": row.get("error_message"),
                }
                for row in unsupported_rows[:20]
            ],
        }

    @staticmethod
    def _unsupported_feature(reason: str, message: str | None = None) -> str:
        text = f"{reason} {message or ''}".lower()
        if "nested" in text:
            return "nested_query"
        if "set operation" in text or reason == "set_operation":
            return "set_operation"
        if "window" in text:
            return "window_function"
        if "having" in text:
            return "having_clause"
        if "case" in text:
            return "case_expression"
        if " or " in f" {text} " or "or filters" in text:
            return "or_filter"
        if "join" in text:
            return "unsupported_join"
        if "select expression" in text:
            return "unsupported_select_expression"
        if "parse" in text:
            return "parse_error"
        if "schema" in text or "unknown" in text or "ambiguous" in text:
            return "schema_mapping_failed"
        if "validation" in text:
            return "validator_failed"
        if "roundtrip" in text or "render" in text:
            return "renderer_failed"
        return reason

    @staticmethod
    def _write_split_files(output: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
        for name in ["train", "validation", "test", "unseen_db_test", "unsupported"]:
            write_jsonl(output / f"generic_ir_{name}.jsonl", splits.get(name, []))

    @staticmethod
    def _schema_dict(schema: Any) -> dict[str, Any]:
        if schema is None:
            return {}
        if isinstance(schema, DatabaseSchema):
            return schema.to_dict()
        return model_dump(schema)

    @staticmethod
    def _serialized_schema(schema: Any) -> str | None:
        if schema is None:
            return None
        if isinstance(schema, DatabaseSchema):
            return schema.serialized_schema
        if isinstance(schema, dict):
            return schema.get("serialized_schema")
        return getattr(schema, "serialized_schema", None)
