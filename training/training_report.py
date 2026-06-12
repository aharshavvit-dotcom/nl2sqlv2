from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


def build_training_report(
    datasets_used: list[str],
    total_loaded: int,
    supported: int,
    unsupported: int,
    examples: list[dict[str, Any]],
    vocabulary_size: int,
    include_schema_text: bool,
) -> dict[str, Any]:
    return {
        "datasets_used": datasets_used,
        "total_loaded_examples": total_loaded,
        "supported_examples": supported,
        "unsupported_examples": unsupported,
        "by_template": dict(Counter(row.get("template_id") for row in examples if row.get("template_id"))),
        "by_dataset": dict(Counter(row.get("dataset_name") for row in examples if row.get("dataset_name"))),
        "vocabulary_size": vocabulary_size,
        "include_schema_text": include_schema_text,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
