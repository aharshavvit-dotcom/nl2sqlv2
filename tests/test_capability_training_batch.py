from __future__ import annotations

from capabilities import ALL_CAPABILITIES
from neural_ir.ir_dataset import capability_label_vector, collate_ir_batch, task_mask_vector


def _item(row: dict) -> dict:
    return {
        "question_ids": [1, 0],
        "schema_ids": [1, 0],
        "question_mask": [1, 0],
        "schema_mask": [1, 0],
        "table_candidate_mask": [1.0],
        "column_candidate_mask": [1.0],
        "metric_column_mask": [1.0],
        "dimension_column_mask": [1.0],
        "date_column_mask": [1.0],
        "filter_column_mask": [1.0],
        "schema_link_scores": [0.0],
        "table_candidate_token_ids": [[0]],
        "column_candidate_token_ids": [[0]],
        "candidate_token_ids": [[0]],
        "relation_type_ids": [[0, 0], [0, 0]],
        "schema_relation_type_ids": [[0, 0], [0, 0]],
        "candidate_relation_type_ids": [[0]],
        "labels": {"intent_label": 0},
        "capability_labels": capability_label_vector(row),
        "task_masks": task_mask_vector(row),
        "raw_example": row,
        "schema_items": {},
        "schema_candidates": {},
        "schema_linking": {},
        "candidate_warnings": [],
    }


def test_capability_training_batch_adds_multilabel_targets_and_masks() -> None:
    row = {
        "required_capabilities": ["AGGREGATION", "GROUP_BY"],
        "task_masks": {"capability": 1, "table": 1, "column": 1, "full_query_ir": 0},
    }
    batch = collate_ir_batch([_item(row)])
    cap_index = {cap.value: index for index, cap in enumerate(ALL_CAPABILITIES)}

    assert batch["capability_labels"].shape == (1, len(ALL_CAPABILITIES))
    assert batch["capability_labels"][0, cap_index["AGGREGATION"]].item() == 1.0
    assert batch["capability_labels"][0, cap_index["GROUP_BY"]].item() == 1.0
    assert batch["task_masks"]["capability"].item() == 1.0
    assert batch["task_masks"]["full_query_ir"].item() == 0.0
