from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from dataset_training.utils import read_jsonl
from .example_index import ExampleIndex
from .pattern_index import PatternIndex
from .schema_index import SchemaIndex


class RAGIndexBuilder:
    def build(self, examples: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
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
        metadata = {"example_count": len(examples), "index_version": "local_rag_v1"}
        (output / "rag_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "output_dir": str(output),
            "example_count": len(examples),
            "files": {
                "example_index": str(output / "example_index.pkl"),
                "schema_index": str(output / "schema_index.pkl"),
                "pattern_index": str(output / "pattern_index.pkl"),
                "metadata": str(output / "rag_metadata.json"),
            },
        }

    def build_from_jsonl(self, input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        return self.build(read_jsonl(input_path), output_dir)
