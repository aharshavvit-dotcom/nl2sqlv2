from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .ir_label_encoder import IRLabelEncoder
from .schema_linearizer import SchemaLinearizer, extract_schema_items, schema_from_example
from .tokenizer import tokenize
from .vocab import Vocabulary


class IRTrainingDataset(Dataset):
    def __init__(
        self,
        path: str,
        vocab: Vocabulary,
        label_encoder: IRLabelEncoder,
        max_question_len: int = 64,
        max_schema_len: int = 256,
        max_tables: int = 64,
        max_columns: int = 256,
    ):
        self.path = Path(path)
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_question_len = max_question_len
        self.max_schema_len = max_schema_len
        self.max_tables = max_tables
        self.max_columns = max_columns
        self.linearizer = SchemaLinearizer()
        self.examples = load_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        schema = schema_from_example(row)
        schema_items = extract_schema_items(schema)
        schema_text = row.get("serialized_schema") or self.linearizer.linearize(schema)
        question_tokens = tokenize(row.get("question", ""))
        schema_tokens = tokenize(schema_text)
        question_ids = self.vocab.encode(question_tokens, self.max_question_len)
        schema_ids = self.vocab.encode(schema_tokens, self.max_schema_len)
        labels = self.label_encoder.encode(row["query_ir"], schema_items)
        labels = self._cap_pointer_labels(labels)
        return {
            "question_ids": question_ids,
            "schema_ids": schema_ids,
            "question_mask": [1 if item != self.vocab.pad_id else 0 for item in question_ids],
            "schema_mask": [1 if item != self.vocab.pad_id else 0 for item in schema_ids],
            "labels": labels,
            "raw_example": row,
            "schema_items": schema_items,
        }

    def _cap_pointer_labels(self, labels: dict[str, int]) -> dict[str, int]:
        capped = dict(labels)
        if capped.get("base_table_index", -1) >= self.max_tables:
            capped["base_table_index"] = -1
        for key in ["metric_column_index", "dimension_column_index", "date_column_index", "filter_column_index"]:
            if capped.get(key, -1) >= self.max_columns:
                capped[key] = -1
        return capped


def collate_ir_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    label_keys = sorted(batch[0]["labels"])
    return {
        "question_ids": torch.tensor([item["question_ids"] for item in batch], dtype=torch.long),
        "schema_ids": torch.tensor([item["schema_ids"] for item in batch], dtype=torch.long),
        "question_mask": torch.tensor([item["question_mask"] for item in batch], dtype=torch.float32),
        "schema_mask": torch.tensor([item["schema_mask"] for item in batch], dtype=torch.float32),
        "labels": {
            key: torch.tensor([item["labels"][key] for item in batch], dtype=torch.long)
            for key in label_keys
        },
        "raw_examples": [item["raw_example"] for item in batch],
        "schema_items": [item["schema_items"] for item in batch],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
