from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from capabilities import ALL_CAPABILITIES, ALL_SAFETY_LABELS
from .candidate_builder import SchemaCandidateBuilder, build_candidate_masks, schema_link_score_vector
from .ir_label_encoder import IRLabelEncoder
from .schema_linearizer import SchemaLinearizer, extract_schema_items, schema_from_example
from .schema_linker import SchemaLinker
from .tokenizer import tokenize
from .vocab import Vocabulary

# Relation types for relation-aware schema attention (matches attention_model.RELATION_TYPES)
RELATION_TYPE_MAP = {
    "same_table": 0, "table_has_column": 1, "column_belongs_to_table": 2,
    "fk_to_pk": 3, "pk_to_fk": 4, "primary_key": 5, "foreign_key_column": 6,
    "same_column_name": 7, "same_data_type": 8, "unrelated": 9,
}
RELATION_UNRELATED = RELATION_TYPE_MAP["unrelated"]
CAPABILITY_INDEX = {capability.value: index for index, capability in enumerate(ALL_CAPABILITIES)}
SAFETY_LABEL_INDEX = {label.value: index for index, label in enumerate(ALL_SAFETY_LABELS)}
TASK_MASK_KEYS = [
    "capability",
    "safety",
    "table",
    "column",
    "aggregation",
    "filter",
    "join_edge",
    "complexity",
    "contrastive_schema_linking",
    "subquery",
    "window",
    "set_operation",
    "full_query_ir",
]
COMPLEXITY_LABELS = [
    "unknown",
    "simple",
    "easy",
    "medium",
    "moderate",
    "hard",
    "challenging",
    "level_1_single_table",
    "level_2_filter_count",
    "level_3_aggregation",
    "level_4_join",
    "level_5_advanced_sql",
]
COMPLEXITY_INDEX = {label: index for index, label in enumerate(COMPLEXITY_LABELS)}


class IRTrainingDataset(Dataset):
    def __init__(
        self,
        path: str,
        vocab: Vocabulary,
        label_encoder: IRLabelEncoder,
        max_question_len: int = 64,
        max_schema_len: int = 256,
        max_candidate_tokens: int = 16,
        max_tables: int = 64,
        max_columns: int = 256,
        max_examples: int | None = None,
        hard_negative_rows: list[dict[str, Any]] | None = None,
    ):
        self.path = Path(path)
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_question_len = max_question_len
        self.max_schema_len = max_schema_len
        self.max_candidate_tokens = max_candidate_tokens
        self.max_tables = max_tables
        self.max_columns = max_columns
        self.linearizer = SchemaLinearizer()
        self.candidate_builder = SchemaCandidateBuilder()
        self.schema_linker = SchemaLinker()
        self.examples = load_jsonl(self.path)
        if max_examples is not None and max_examples > 0:
            self.examples = self.examples[:max_examples]
        self.hard_negative_by_example = _index_hard_negatives(hard_negative_rows or [])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        schema = schema_from_example(row)
        schema_items = extract_schema_items(schema)
        schema_text = row.get("serialized_schema") or self.linearizer.linearize(schema)
        question = row.get("question", "")
        question_tokens = tokenize(question)
        schema_tokens = tokenize(schema_text)
        question_ids = self.vocab.encode(question_tokens, self.max_question_len)
        schema_ids = self.vocab.encode(schema_tokens, self.max_schema_len)
        candidates = self.candidate_builder.build_candidates(schema, question)
        link_result = self.schema_linker.link(question, candidates)
        candidate_masks = build_candidate_masks(candidates, self.max_tables, self.max_columns)
        link_scores = schema_link_score_vector(link_result, self.max_columns)
        table_candidate_token_ids = self._candidate_token_ids(candidates.get("tables", []), self.max_tables)
        column_candidate_token_ids = self._candidate_token_ids(candidates.get("columns", []), self.max_columns)
        labels = self.label_encoder.encode(row["query_ir"], schema_items)
        labels = self._add_hard_negative_labels(row, schema_items, labels)
        labels = self._cap_pointer_labels(labels)
        labels["span"] = self._get_span_labels(row, question)
        labels["complexity_label"] = complexity_label(row)
        self._force_gold_masks(candidate_masks, labels)
        candidate_metadata = build_candidate_metadata(candidates, schema_items, self.max_tables)
        unified_candidate_token_ids = self._unified_candidate_token_ids(candidate_metadata, self.max_tables + self.max_columns)
        
        return {
            "question_ids": question_ids,
            "schema_ids": schema_ids,
            "question_mask": [1 if item != self.vocab.pad_id else 0 for item in question_ids],
            "schema_mask": [1 if item != self.vocab.pad_id else 0 for item in schema_ids],
            "table_candidate_mask": candidate_masks["table_candidate_mask"],
            "column_candidate_mask": candidate_masks["column_candidate_mask"],
            "metric_column_mask": candidate_masks["metric_column_mask"],
            "dimension_column_mask": candidate_masks["dimension_column_mask"],
            "date_column_mask": candidate_masks["date_column_mask"],
            "filter_column_mask": candidate_masks["filter_column_mask"],
            "schema_link_scores": link_scores,
            "table_candidate_token_ids": table_candidate_token_ids,
            "column_candidate_token_ids": column_candidate_token_ids,
            "candidate_token_ids": unified_candidate_token_ids,
            "relation_type_ids": build_question_schema_relation_type_ids(
                schema_items,
                schema_tokens,
                self.max_question_len,
                self.max_schema_len,
            ),
            "schema_relation_type_ids": build_schema_relation_type_ids(
                schema_items,
                schema_tokens,
                self.max_schema_len,
            ),
            "candidate_relation_type_ids": build_candidate_pairwise_relation_matrix(
                candidate_metadata,
                self.max_tables + self.max_columns,
            ),
            "labels": labels,
            "capability_labels": capability_label_vector(row),
            "safety_labels": safety_label_vector(row),
            "task_masks": task_mask_vector(row),
            "raw_example": row,
            "schema_items": schema_items,
            "schema_candidates": candidates,
            "schema_linking": link_result,
            "candidate_warnings": candidate_masks.get("candidate_warnings", []),
        }

    def _cap_pointer_labels(self, labels: dict[str, int]) -> dict[str, int]:
        capped = dict(labels)
        if capped.get("base_table_index", -1) >= self.max_tables:
            capped["base_table_index"] = -1
        for key in ["metric_column_index", "dimension_column_index", "date_column_index", "filter_column_index"]:
            if capped.get(key, -1) >= self.max_columns:
                capped[key] = -1
        for key in [
            "negative_base_table_index",
            "negative_metric_column_index",
            "negative_dimension_column_index",
            "negative_date_column_index",
            "negative_filter_column_index",
        ]:
            limit = self.max_tables if key == "negative_base_table_index" else self.max_columns
            if capped.get(key, -1) >= limit:
                capped[key] = -1
        return capped

    def _add_hard_negative_labels(
        self,
        row: dict[str, Any],
        schema_items: dict[str, Any],
        labels: dict[str, int],
    ) -> dict[str, int]:
        enriched = {
            **labels,
            "negative_base_table_index": -1,
            "negative_metric_column_index": -1,
            "negative_dimension_column_index": -1,
            "negative_date_column_index": -1,
            "negative_filter_column_index": -1,
        }
        example_id = str(row.get("example_id") or "")
        negative_row = self.hard_negative_by_example.get(example_id)
        negative_ir = (negative_row or {}).get("negative_query_ir") or (negative_row or {}).get("query_ir")
        if not isinstance(negative_ir, dict):
            return enriched
        negative_labels = self.label_encoder.encode(negative_ir, schema_items)
        enriched["negative_base_table_index"] = int(negative_labels.get("base_table_index", -1))
        enriched["negative_metric_column_index"] = int(negative_labels.get("metric_column_index", -1))
        enriched["negative_dimension_column_index"] = int(negative_labels.get("dimension_column_index", -1))
        enriched["negative_date_column_index"] = int(negative_labels.get("date_column_index", -1))
        enriched["negative_filter_column_index"] = int(negative_labels.get("filter_column_index", -1))
        return enriched

    @staticmethod
    def _force_gold_masks(candidate_masks: dict[str, list[float]], labels: dict[str, int]) -> None:
        pointers = {
            "table_candidate_mask": "base_table_index",
            "column_candidate_mask": None,
            "metric_column_mask": "metric_column_index",
            "dimension_column_mask": "dimension_column_index",
            "date_column_mask": "date_column_index",
            "filter_column_mask": "filter_column_index",
        }
        for mask_key, label_key in pointers.items():
            if not label_key:
                continue
            index = int(labels.get(label_key, -1))
            mask = candidate_masks.get(mask_key) or []
            if 0 <= index < len(mask):
                mask[index] = 1.0

    def _candidate_token_ids(self, candidates: list[dict[str, Any]], max_candidates: int) -> list[list[int]]:
        rows = [[self.vocab.pad_id] * self.max_candidate_tokens for _ in range(max_candidates)]
        for candidate in candidates:
            index = int(candidate.get("index", -1))
            if not (0 <= index < max_candidates):
                continue
            tokens = list(candidate.get("tokens") or [])
            if not tokens:
                tokens = tokenize(str(candidate.get("display") or ""))
            rows[index] = self.vocab.encode(tokens, self.max_candidate_tokens)
        return rows

    def _unified_candidate_token_ids(self, candidate_metadata: list[dict[str, Any]], max_candidates: int) -> list[list[int]]:
        rows = [[self.vocab.pad_id] * self.max_candidate_tokens for _ in range(max_candidates)]
        for candidate in candidate_metadata:
            index = int(candidate.get("candidate_index", -1))
            if not (0 <= index < max_candidates):
                continue
            name = candidate.get("column_name") or candidate.get("table_name") or ""
            tokens = tokenize(str(name))
            rows[index] = self.vocab.encode(tokens, self.max_candidate_tokens)
        return rows

    def _get_span_labels(self, row: dict[str, Any], question: str) -> list[int]:
        import re
        token_re = re.compile(r"[a-z0-9_]+")
        span_labels = [0] * self.max_question_len
        query_ir = row.get("query_ir")
        if not isinstance(query_ir, dict):
            return span_labels
            
        filters = query_ir.get("filters") or []
        values = []
        for filt in filters:
            if isinstance(filt, dict) and "value" in filt:
                val = filt["value"]
                if val is not None:
                    if isinstance(val, list):
                        values.extend(val)
                    else:
                        values.append(val)
                        
        if not values:
            return span_labels
            
        token_spans = []
        for m in token_re.finditer(question.lower()):
            token_spans.append((m.start(), m.end()))
            
        for val in values:
            val_str = str(val).strip()
            if not val_str:
                continue
                
            patterns = [re.escape(val_str)]
            if isinstance(val, (int, float)):
                patterns.append(re.escape(f"{val:,}"))
                
            for pattern in patterns:
                for m in re.finditer(pattern, question, re.IGNORECASE):
                    val_start, val_end = m.start(), m.end()
                    overlapping_indices = []
                    for idx, (t_start, t_end) in enumerate(token_spans):
                        if max(t_start, val_start) < min(t_end, val_end):
                            overlapping_indices.append(idx)
                    for idx in overlapping_indices:
                        if 0 <= idx < self.max_question_len:
                            span_labels[idx] = 1
                            
        return span_labels


def collate_ir_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    label_keys = sorted(batch[0]["labels"])
    return {
        "question_ids": torch.tensor([item["question_ids"] for item in batch], dtype=torch.long),
        "schema_ids": torch.tensor([item["schema_ids"] for item in batch], dtype=torch.long),
        "question_mask": torch.tensor([item["question_mask"] for item in batch], dtype=torch.float32),
        "schema_mask": torch.tensor([item["schema_mask"] for item in batch], dtype=torch.float32),
        "table_candidate_mask": torch.tensor([item["table_candidate_mask"] for item in batch], dtype=torch.float32),
        "column_candidate_mask": torch.tensor([item["column_candidate_mask"] for item in batch], dtype=torch.float32),
        "metric_column_mask": torch.tensor([item["metric_column_mask"] for item in batch], dtype=torch.float32),
        "dimension_column_mask": torch.tensor([item["dimension_column_mask"] for item in batch], dtype=torch.float32),
        "date_column_mask": torch.tensor([item["date_column_mask"] for item in batch], dtype=torch.float32),
        "filter_column_mask": torch.tensor([item["filter_column_mask"] for item in batch], dtype=torch.float32),
        "schema_link_scores": torch.tensor([item["schema_link_scores"] for item in batch], dtype=torch.float32),
        "table_candidate_token_ids": torch.tensor([item["table_candidate_token_ids"] for item in batch], dtype=torch.long),
        "column_candidate_token_ids": torch.tensor([item["column_candidate_token_ids"] for item in batch], dtype=torch.long),
        "candidate_token_ids": torch.tensor([item["candidate_token_ids"] for item in batch], dtype=torch.long),
        "relation_type_ids": torch.tensor([item["relation_type_ids"] for item in batch], dtype=torch.long),
        "schema_relation_type_ids": torch.tensor([item["schema_relation_type_ids"] for item in batch], dtype=torch.long),
        "candidate_relation_type_ids": torch.tensor([item["candidate_relation_type_ids"] for item in batch], dtype=torch.long),
        "labels": {
            key: torch.tensor([item["labels"][key] for item in batch], dtype=torch.long)
            for key in label_keys
        },
        "capability_labels": torch.tensor([item["capability_labels"] for item in batch], dtype=torch.float32),
        "safety_labels": torch.tensor([item["safety_labels"] for item in batch], dtype=torch.float32),
        "task_masks": {
            key: torch.tensor([item["task_masks"].get(key, 0.0) for item in batch], dtype=torch.float32)
            for key in TASK_MASK_KEYS
        },
        "raw_examples": [item["raw_example"] for item in batch],
        "schema_items": [item["schema_items"] for item in batch],
        "schema_candidates": [item["schema_candidates"] for item in batch],
        "schema_linking": [item["schema_linking"] for item in batch],
        "candidate_warnings": [item["candidate_warnings"] for item in batch],
    }


def build_question_schema_relation_type_ids(
    schema_items: dict[str, Any],
    schema_tokens: list[str],
    max_question_len: int,
    max_schema_len: int,
) -> list[list[int]]:
    """Build [question_len, schema_len] role-bias IDs for question-schema attention."""
    roles = [
        _schema_token_role(entity)
        for entity in _schema_token_entities(schema_items, schema_tokens, max_schema_len)
    ]
    return [list(roles) for _ in range(max_question_len)]


def build_schema_relation_type_ids(
    schema_items: dict[str, Any],
    schema_tokens: list[str],
    max_schema_len: int,
) -> list[list[int]]:
    """Build [schema_len, schema_len] pairwise schema relation IDs."""
    entities = _schema_token_entities(schema_items, schema_tokens, max_schema_len)
    return [
        [_pairwise_relation(left, right) for right in entities]
        for left in entities
    ]


def _schema_token_entities(
    schema_items: dict[str, Any],
    schema_tokens: list[str],
    max_schema_len: int,
) -> list[dict[str, Any] | None]:
    table_by_token: dict[str, dict[str, Any]] = {}
    for table in schema_items.get("tables", []):
        for token in tokenize(str(table).replace("_", " ")):
            table_by_token.setdefault(token, {"kind": "table", "table": str(table)})

    column_by_token: dict[str, dict[str, Any]] = {}
    for column in schema_items.get("columns", []):
        entity = {
            "kind": "column",
            "table": str(column.get("table") or ""),
            "column": str(column.get("column") or ""),
            "type": str(column.get("type") or ""),
            "primary_key": bool(column.get("primary_key", False)),
            "foreign_key": bool(column.get("foreign_key", column.get("is_fk", False))),
            "foreign_key_target": _normalize_fk_target(column.get("foreign_key_target")),
        }
        for token in tokenize(entity["column"].replace("_", " ")):
            column_by_token.setdefault(token, entity)

    entities: list[dict[str, Any] | None] = [None] * max_schema_len
    for index, token in enumerate(schema_tokens[:max_schema_len]):
        normalized = str(token).lower()
        entities[index] = column_by_token.get(normalized) or table_by_token.get(normalized)
    return entities


def _schema_token_role(entity: dict[str, Any] | None) -> int:
    if not entity:
        return RELATION_UNRELATED
    if entity.get("kind") == "table":
        return RELATION_TYPE_MAP["same_table"]
    if entity.get("primary_key"):
        return RELATION_TYPE_MAP["primary_key"]
    if entity.get("foreign_key"):
        return RELATION_TYPE_MAP["foreign_key_column"]
    if entity.get("table"):
        return RELATION_TYPE_MAP["column_belongs_to_table"]
    return RELATION_UNRELATED


def _pairwise_relation(left: dict[str, Any] | None, right: dict[str, Any] | None) -> int:
    if not left or not right:
        return RELATION_UNRELATED
    left_kind = left.get("kind")
    right_kind = right.get("kind")
    if left_kind == "table" and right_kind == "table":
        return RELATION_TYPE_MAP["same_table"] if left.get("table") == right.get("table") else RELATION_UNRELATED
    if left_kind == "table" and right_kind == "column":
        return RELATION_TYPE_MAP["table_has_column"] if left.get("table") == right.get("table") else RELATION_UNRELATED
    if left_kind == "column" and right_kind == "table":
        return RELATION_TYPE_MAP["column_belongs_to_table"] if left.get("table") == right.get("table") else RELATION_UNRELATED
    if left_kind == "column" and right_kind == "column":
        if _fk_points_to(left, right):
            return RELATION_TYPE_MAP["fk_to_pk"]
        if _fk_points_to(right, left):
            return RELATION_TYPE_MAP["pk_to_fk"]
        if left.get("table") == right.get("table"):
            if left.get("column") == right.get("column") and left.get("primary_key"):
                return RELATION_TYPE_MAP["primary_key"]
            if left.get("column") == right.get("column") and left.get("foreign_key"):
                return RELATION_TYPE_MAP["foreign_key_column"]
            return RELATION_TYPE_MAP["same_table"]
        if left.get("column") == right.get("column"):
            return RELATION_TYPE_MAP["same_column_name"]
        if left.get("type") and left.get("type") == right.get("type"):
            return RELATION_TYPE_MAP["same_data_type"]
    return RELATION_UNRELATED


def _fk_points_to(left: dict[str, Any], right: dict[str, Any]) -> bool:
    target = _normalize_fk_target(left.get("foreign_key_target"))
    if not target:
        return False
    return target.get("table") == right.get("table") and target.get("column") == right.get("column")


def _normalize_fk_target(raw: Any) -> dict[str, str] | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        table = raw.get("table") or raw.get("to_table") or raw.get("referred_table")
        column = raw.get("column") or raw.get("to_column") or raw.get("referred_column")
        if table and column:
            return {"table": str(table), "column": str(column)}
    if isinstance(raw, str) and "." in raw:
        table, column = raw.split(".", 1)
        return {"table": table, "column": column}
    return None


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


def capability_label_vector(row: dict[str, Any]) -> list[float]:
    values = row.get("required_capabilities")
    if values is None:
        values = ((row.get("capability_annotation") or {}).get("required_capabilities") or [])
    labels = [0.0] * len(CAPABILITY_INDEX)
    for value in values or []:
        name = str(value)
        if name in CAPABILITY_INDEX:
            labels[CAPABILITY_INDEX[name]] = 1.0
    return labels


def safety_label_vector(row: dict[str, Any]) -> list[float]:
    values = row.get("safety_labels")
    if values is None:
        values = ((row.get("capability_annotation") or {}).get("safety_labels") or [])
    labels = [0.0] * len(SAFETY_LABEL_INDEX)
    for value in values or []:
        name = str(value)
        if name in SAFETY_LABEL_INDEX:
            labels[SAFETY_LABEL_INDEX[name]] = 1.0
    return labels


def complexity_label(row: dict[str, Any]) -> int:
    raw = row.get("complexity")
    if raw is None:
        raw = ((row.get("sql_features") or {}).get("complexity"))
    if raw is None:
        raw = ((row.get("capability_annotation") or {}).get("complexity"))
    normalized = str(raw or "unknown").strip().lower()
    return COMPLEXITY_INDEX.get(normalized, COMPLEXITY_INDEX["unknown"])


def task_mask_vector(row: dict[str, Any]) -> dict[str, float]:
    masks = row.get("task_masks")
    if masks is None:
        masks = ((row.get("capability_annotation") or {}).get("task_masks") or {})
    return {key: float((masks or {}).get(key, 0.0) or 0.0) for key in TASK_MASK_KEYS}


def _index_hard_negatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in indexed:
            continue
        negative_ir = row.get("negative_query_ir") or row.get("query_ir")
        if isinstance(negative_ir, dict):
            indexed[example_id] = row
    return indexed


# ---------------------------------------------------------------------------
# Phase 7: Candidate-level graph encoding
# ---------------------------------------------------------------------------


def build_candidate_metadata(
    candidates: dict[str, Any],
    schema_items: dict[str, Any],
    max_tables: int,
) -> list[dict[str, Any]]:
    """Build stable candidate-level metadata with canonical IDs.

    Each entry contains:
      candidate_index: unified index (tables=index, columns=max_tables+index)
      candidate_id: stable ID like "table:users" or "column:users.id"
      candidate_type: "table" or "column"
      table_name, column_name, data_type, is_primary_key, is_foreign_key, foreign_key_target
    """
    metadata: list[dict[str, Any]] = []
    # Tables first
    for table_cand in candidates.get("tables", []):
        table_name = str(table_cand.get("table") or table_cand.get("display") or "")
        orig_index = int(table_cand.get("index", -1))
        metadata.append({
            "candidate_id": f"table:{table_name}",
            "candidate_type": "table",
            "candidate_index": orig_index if orig_index >= 0 else -1,
            "original_table_index": orig_index,
            "original_column_index": None,
            "table_name": table_name,
            "column_name": None,
            "data_type": None,
            "is_primary_key": False,
            "is_foreign_key": False,
            "foreign_key_target": None,
        })
    # Columns
    columns_by_name: dict[str, dict[str, Any]] = {}
    for col in schema_items.get("columns", []):
        key = f"{col.get('table', '')}.{col.get('column', '')}"
        columns_by_name[key] = col
    for col_cand in candidates.get("columns", []):
        table_name = str(col_cand.get("table") or "")
        column_name = str(col_cand.get("column") or "")
        key = f"{table_name}.{column_name}"
        schema_col = columns_by_name.get(key, {})
        orig_index = int(col_cand.get("index", -1))
        metadata.append({
            "candidate_id": f"column:{table_name}.{column_name}",
            "candidate_type": "column",
            "candidate_index": (max_tables + orig_index) if orig_index >= 0 else -1,
            "original_table_index": None,
            "original_column_index": orig_index,
            "table_name": table_name,
            "column_name": column_name,
            "data_type": str(schema_col.get("type") or col_cand.get("type") or ""),
            "is_primary_key": bool(schema_col.get("primary_key", False)),
            "is_foreign_key": bool(schema_col.get("foreign_key", schema_col.get("is_fk", False))),
            "foreign_key_target": _normalize_fk_target(
                schema_col.get("foreign_key_target")
            ),
        })
    return metadata


def build_candidate_pairwise_relation_matrix(
    candidate_metadata: list[dict[str, Any]],
    max_candidates: int,
) -> list[list[int]]:
    """Build [max_candidates, max_candidates] pairwise relation matrix using stable candidate IDs.

    This operates at candidate-level granularity (not token-level), avoiding
    fragile multi-token matching and duplicate-column-name issues.
    """
    matrix = [[RELATION_UNRELATED] * max_candidates for _ in range(max_candidates)]
    for left in candidate_metadata:
        li = left.get("candidate_index", -1)
        if not (0 <= li < max_candidates):
            continue
        for right in candidate_metadata:
            ri = right.get("candidate_index", -1)
            if not (0 <= ri < max_candidates):
                continue
            matrix[li][ri] = _candidate_pairwise_relation(left, right)
    return matrix


def _candidate_pairwise_relation(
    left: dict[str, Any],
    right: dict[str, Any],
) -> int:
    """Compute relation type between two candidates using stable metadata."""
    left_type = left.get("candidate_type", "")
    right_type = right.get("candidate_type", "")

    if left_type == "table" and right_type == "table":
        return (
            RELATION_TYPE_MAP["same_table"]
            if left.get("table_name") == right.get("table_name")
            else RELATION_UNRELATED
        )
    if left_type == "table" and right_type == "column":
        return (
            RELATION_TYPE_MAP["table_has_column"]
            if left.get("table_name") == right.get("table_name")
            else RELATION_UNRELATED
        )
    if left_type == "column" and right_type == "table":
        return (
            RELATION_TYPE_MAP["column_belongs_to_table"]
            if left.get("table_name") == right.get("table_name")
            else RELATION_UNRELATED
        )
    if left_type == "column" and right_type == "column":
        # FK→PK
        left_fk = left.get("foreign_key_target")
        right_fk = right.get("foreign_key_target")
        if left_fk and left_fk.get("table") == right.get("table_name") and left_fk.get("column") == right.get("column_name"):
            return RELATION_TYPE_MAP["fk_to_pk"]
        if right_fk and right_fk.get("table") == left.get("table_name") and right_fk.get("column") == left.get("column_name"):
            return RELATION_TYPE_MAP["pk_to_fk"]
        # Same table
        if left.get("table_name") == right.get("table_name"):
            if left.get("candidate_id") == right.get("candidate_id"):
                if left.get("is_primary_key"):
                    return RELATION_TYPE_MAP["primary_key"]
                if left.get("is_foreign_key"):
                    return RELATION_TYPE_MAP["foreign_key_column"]
            return RELATION_TYPE_MAP["same_table"]
        # Same column name across tables
        if left.get("column_name") == right.get("column_name"):
            return RELATION_TYPE_MAP["same_column_name"]
        # Same data type
        if left.get("data_type") and left.get("data_type") == right.get("data_type"):
            return RELATION_TYPE_MAP["same_data_type"]
    return RELATION_UNRELATED
