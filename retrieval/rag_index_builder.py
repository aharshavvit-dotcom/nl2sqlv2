from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from collections import Counter

import joblib

from dataset_training.utils import read_jsonl
from .example_index import ExampleIndex
from .pattern_index import PatternIndex
from .schema_index import SchemaIndex
from .artifact_compatibility import build_sklearn_metadata, write_sklearn_metadata


class RAGIndexBuilder:
    def build(
        self,
        examples: list[dict[str, Any]],
        output_dir: str | Path,
        source_train_file: str | Path | None = None,
    ) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        example_index = ExampleIndex()
        schema_index = SchemaIndex()
        pattern_index = PatternIndex()
        example_index.build(examples)
        schema_index.build(examples)
        pattern_index.build(examples)
        example_index.save(str(output / "example_index.pkl"))
        joblib.dump(schema_index, output / "schema_index.pkl")
        joblib.dump(pattern_index, output / "pattern_index.pkl")
        by_dataset = Counter(str(row.get("dataset_name") or row.get("dataset") or "unknown") for row in examples)
        intent_distribution = Counter(str(row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown") for row in examples)
        sql_complexity_distribution = Counter(
            str(row.get("complexity") or (row.get("sql_features") or {}).get("complexity") or "unknown")
            for row in examples
        )
        metadata = {
            "example_count": len(examples),
            "index_version": "local_rag_v1",
            "by_dataset": dict(by_dataset),
            "intent_distribution": dict(intent_distribution),
            "sql_complexity_distribution": dict(sql_complexity_distribution),
        }
        sklearn_metadata = build_sklearn_metadata(
            artifact_types=["tfidf_vectorizer", "tfidf_matrix", "schema_index", "pattern_index"],
            source_path=source_train_file,
            config={"index_version": metadata["index_version"]},
        )
        write_sklearn_metadata(output, sklearn_metadata)
        metadata["sklearn_artifact_metadata"] = sklearn_metadata
        (output / "rag_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest = {
            "source_train_file": str(source_train_file or ""),
            "total_examples": len(examples),
            "by_dataset": dict(by_dataset),
            "intent_distribution": dict(intent_distribution),
            "sql_complexity_distribution": dict(sql_complexity_distribution),
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "schema_index_built": (output / "schema_index.pkl").exists(),
            "example_index_built": (output / "example_index.pkl").exists(),
            "pattern_index_built": (output / "pattern_index.pkl").exists(),
            "sklearn_version": sklearn_metadata["sklearn_version"],
            "sklearn_artifact_metadata": sklearn_metadata,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "output_dir": str(output),
            "example_count": len(examples),
            "by_dataset": dict(by_dataset),
            "files": {
                "example_index": str(output / "example_index.pkl"),
                "schema_index": str(output / "schema_index.pkl"),
                "pattern_index": str(output / "pattern_index.pkl"),
                "metadata": str(output / "rag_metadata.json"),
                "manifest": str(output / "manifest.json"),
                "sklearn_artifact_metadata": str(output / "sklearn_artifact_metadata.json"),
            },
        }

    def build_from_jsonl(self, input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        return self.build(read_jsonl(input_path), output_dir, source_train_file=input_path)
