from __future__ import annotations

from enum import Enum
import hashlib
import json
import re
from typing import Any
from rapidfuzz import fuzz

from inference.runtime_schema_context import RuntimeSchemaContext


class ValueIndexMode(str, Enum):
    DISABLED = "disabled"
    METADATA_ONLY = "metadata_only"
    APPROVED_DOMAIN_VALUES = "approved_domain_values"
    CONTROLLED_SAMPLE = "controlled_sample"


SENSITIVE_PATTERNS = [
    r"pass(word)?",
    r"ssn",
    r"aadhaar",
    r"credit",
    r"card",
    r"cvv",
    r"token",
    r"key",
    r"email",
    r"phone",
    r"addr",
    r"birth",
    r"dob",
    r"medical",
    r"comment",
    r"notes",
]


class SchemaValueIndex:
    def __init__(
        self,
        schema_context: RuntimeSchemaContext,
        mode: ValueIndexMode = ValueIndexMode.APPROVED_DOMAIN_VALUES,
        max_rows_per_column: int = 500,
        max_distinct_values: int = 200,
        max_value_length: int = 128,
        exclude_high_cardinality_ratio: float = 0.25,
    ):
        self.schema_context = schema_context
        self.mode = mode
        self.max_rows_per_column = max_rows_per_column
        self.max_distinct_values = max_distinct_values
        self.max_value_length = max_value_length
        self.exclude_high_cardinality_ratio = exclude_high_cardinality_ratio

        self.schema_fingerprint = self._compute_schema_fingerprint()
        self.index: dict[str, list[tuple[str, str]]] = {}

        self._build_index()

    def _compute_schema_fingerprint(self) -> str:
        tbls = sorted(self.schema_context.get_tables())
        hasher = hashlib.sha256()
        for t in tbls:
            hasher.update(t.encode("utf-8"))
            cols = sorted(self.schema_context.get_table_columns(t))
            for c in cols:
                hasher.update(c.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _is_column_sensitive(self, table: str, column: str) -> bool:
        info = self.schema_context.column_info(table, column)
        if info.get("is_sensitive"):
            return True
        col_lower = column.lower()
        if any(re.search(pat, col_lower) for pat in SENSITIVE_PATTERNS):
            return True
        return False

    def _build_index(self) -> None:
        if self.mode == ValueIndexMode.DISABLED:
            return

        for qualified in self.schema_context.get_columns():
            table, column = qualified.split(".", 1)
            if self._is_column_sensitive(table, column):
                continue

            info = self.schema_context.column_info(table, column)
            samples = info.get("sample_values") or []

            if self.mode == ValueIndexMode.CONTROLLED_SAMPLE:
                samples = samples[: self.max_distinct_values]

            for val in samples:
                val_str = str(val).strip()
                if len(val_str) > self.max_value_length:
                    continue
                val_norm = self._normalize_val(val_str)
                if val_norm:
                    self.index.setdefault(val_norm, []).append((table, column))

    def _normalize_val(self, val: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", val.lower()).strip().split())

    def lookup_value(self, value_str: str) -> list[dict[str, Any]]:
        """Look up matching columns for a filter value."""
        if self.mode == ValueIndexMode.DISABLED:
            return []

        value_lower = value_str.lower().strip()
        value_norm = self._normalize_val(value_str)

        candidates = []
        matching_columns = self.index.get(value_norm, [])
        for table, column in matching_columns:
            candidates.append({
                "table": table,
                "column": f"{table}.{column}",
                "score": 0.94,
                "signals": {"exact_value_match": 1.0, "type_compatibility": 1.0},
            })

        if not candidates:
            for val_norm, cols in self.index.items():
                ratio = fuzz.ratio(value_norm, val_norm) / 100.0
                if ratio >= 0.85:
                    score = round(0.80 * ratio, 4)
                    for table, column in cols:
                        candidates.append({
                            "table": table,
                            "column": f"{table}.{column}",
                            "score": score,
                            "signals": {"fuzzy_value_match": ratio, "type_compatibility": 1.0},
                        })

        return candidates
