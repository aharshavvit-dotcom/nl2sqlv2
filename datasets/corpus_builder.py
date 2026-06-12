from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.dataset_paths import PROCESSED_DATA_DIR, ensure_dataset_dirs, parse_dataset_list

from .dataset_loader import DatasetLoader
from .models import DatabaseSchema, Text2SQLExample, TrainingCorpusStats
from .schema_normalizer import SchemaNormalizer
from .sql_feature_extractor import SQLFeatureExtractor
from .sql_pattern_classifier import SQLPatternClassifier


class CorpusBuilder:
    def __init__(
        self,
        loader: DatasetLoader | None = None,
        output_dir: Path = PROCESSED_DATA_DIR,
    ):
        self.loader = loader or DatasetLoader()
        self.output_dir = output_dir
        self.extractor = SQLFeatureExtractor()
        self.classifier = SQLPatternClassifier()
        self.normalizer = SchemaNormalizer()

    def build_corpus(
        self,
        dataset_names: list[str],
        max_examples: int | None = None,
        include_schema_text: bool = False,
    ) -> dict[str, Any]:
        ensure_dataset_dirs()
        requested = parse_dataset_list(dataset_names)
        all_examples: list[Text2SQLExample] = []
        schemas: dict[str, DatabaseSchema] = {}

        for dataset_name in requested:
            examples, dataset_schemas = self.loader.load(dataset_name, max_examples=max_examples)
            if max_examples is not None:
                examples = examples[:max_examples]
            all_examples.extend(examples)
            schemas.update(dataset_schemas)

        processed = self.process_examples(all_examples, schemas, include_schema_text=include_schema_text)
        stats = self._stats(processed)
        return {
            "examples": processed,
            "schemas": schemas,
            "schema_registry": self.normalizer.build_schema_registry(processed, schemas),
            "stats": stats,
        }

    def process_examples(
        self,
        examples: list[Text2SQLExample],
        schemas: dict[str, DatabaseSchema],
        include_schema_text: bool = False,
    ) -> list[Text2SQLExample]:
        processed: list[Text2SQLExample] = []
        for example in examples:
            features = self.extractor.extract(example.sql)
            classification = self.classifier.classify(example.sql, features)
            schema = schemas.get(example.db_id)
            schema_text = schema.serialized_schema if schema else None
            if schema and schema_text:
                example.schema = schema.to_dict()
                example.tables = example.tables or list(schema.tables)
                example.columns = example.columns or self._columns_from_schema(schema)
            example.sql_features = features
            example.template_id = classification["template_id"]
            example.intent = classification["intent"]
            example.is_supported = bool(classification["is_supported"])
            example.unsupported_reason = classification["unsupported_reason"]
            example.extracted_slots = self._basic_slots(features)
            if include_schema_text and schema_text:
                example.extracted_slots["serialized_schema"] = schema_text
            processed.append(example)
        return processed

    def save_outputs(self, processed_payload: dict[str, Any], output_dir: Path | None = None) -> None:
        output = output_dir or self.output_dir
        output.mkdir(parents=True, exist_ok=True)
        examples: list[Text2SQLExample] = processed_payload["examples"]
        supported = [example for example in examples if example.is_supported]
        unsupported = [example for example in examples if not example.is_supported]
        self._write_jsonl(output / "unified_examples.jsonl", [self._dump_model(item) for item in examples])
        self._write_jsonl(output / "supported_examples.jsonl", [self._dump_model(item) for item in supported])
        self._write_jsonl(output / "unsupported_examples.jsonl", [self._dump_model(item) for item in unsupported])
        self._write_jsonl(output / "schema_registry.jsonl", processed_payload["schema_registry"])
        self._write_json(output / "dataset_stats.json", processed_payload["stats"].to_dict())

    @staticmethod
    def _basic_slots(features: dict[str, Any]) -> dict[str, Any]:
        aggregation = (features.get("aggregation_expressions") or [{}])[0]
        return {
            "metric": aggregation.get("column"),
            "dimension": (features.get("group_by") or [None])[0],
            "entity": (features.get("tables") or [None])[0],
            "limit": features.get("limit"),
            "selected_columns": features.get("selected_columns", []),
            "filter_columns": [item.get("expression") for item in features.get("where_conditions", [])],
        }

    @staticmethod
    def _stats(examples: list[Text2SQLExample]) -> TrainingCorpusStats:
        by_dataset = Counter(example.dataset_name for example in examples)
        by_template = Counter(example.template_id for example in examples if example.template_id)
        by_split = Counter(example.split for example in examples)
        unsupported = Counter(example.unsupported_reason for example in examples if example.unsupported_reason)
        supported_count = sum(1 for example in examples if example.is_supported)
        return TrainingCorpusStats(
            total_examples=len(examples),
            supported_examples=supported_count,
            unsupported_examples=len(examples) - supported_count,
            by_dataset=dict(by_dataset),
            by_template=dict(by_template),
            by_split=dict(by_split),
            unsupported_reasons=dict(unsupported),
        )

    @staticmethod
    def _columns_from_schema(schema: DatabaseSchema) -> list[str]:
        columns: list[str] = []
        for table_info in schema.tables.values():
            for column in SchemaNormalizer._columns_from_table_info(table_info):
                columns.append(column)
        return columns

    @staticmethod
    def _dump_model(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        if hasattr(model, "dict"):
            return model.dict()
        return dict(model)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
