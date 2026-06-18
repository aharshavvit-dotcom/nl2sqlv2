from __future__ import annotations

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
    ) -> dict[str, Any]:
        output = Path(output_dir)
        artifacts = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        requested = [normalize_dataset_name(item) for item in datasets]
        registry_report = self.dataset_registry.validate_dataset_presence(requested)
        examples, schemas = self._load_examples(requested, registry_report, max_examples=max_examples)

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

        return {
            "output_dir": str(output),
            "artifact_dir": str(artifacts),
            "split_report": split_report,
            "leakage_report": leakage_report,
            "corpus_quality_report": quality_report,
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
    ) -> tuple[list[Text2SQLExample], dict[str, DatabaseSchema]]:
        if hasattr(self.dataset_registry, "load_examples"):
            return self.dataset_registry.load_examples(datasets, max_examples=max_examples)  # type: ignore[attr-defined]

        loader = DatasetLoader(raw_root=self.dataset_registry.root_dir)
        all_examples: list[Text2SQLExample] = []
        schemas: dict[str, DatabaseSchema] = {}
        for dataset_name in datasets:
            if not registry_report.get(dataset_name, {}).get("available"):
                continue
            remaining = None if max_examples is None else max(max_examples - len(all_examples), 0)
            if remaining == 0:
                break
            try:
                examples, dataset_schemas = loader.load(dataset_name, max_examples=remaining)
            except Exception:
                continue
            if remaining is not None:
                examples = examples[:remaining]
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
            "dataset_name": example.dataset_name,
            "db_id": example.db_id,
            "question": example.question,
            "source_sql": example.sql,
            "unsupported_reason": reason or result.get("unsupported_reason") or "unsupported",
            "error_message": result.get("error_message"),
            "metadata": {
                "difficulty": example.difficulty,
                "source_file": example.source_file,
                "split": example.split,
                "warnings": result.get("warnings", []),
            },
        }

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
