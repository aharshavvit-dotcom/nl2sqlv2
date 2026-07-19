"""
Purpose: Verifies capabilities unit behaviour consolidated from fragmented test files.
Required because: Capability extraction, taxonomy, artifacts and training parity are one capability-data contract.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_capability_artifact_schema.py
import pytest
from pydantic import ValidationError

from capabilities import CapabilityAnnotation, SQLCapabilityExtractor


def test_capability_artifact_schema_rejects_unknown_fields() -> None:
    payload = SQLCapabilityExtractor().extract("SELECT id FROM users").model_dump(mode="json")
    payload["unknown_field"] = "not allowed"

    with pytest.raises(ValidationError):
        CapabilityAnnotation.model_validate(payload)


def test_capability_artifact_schema_roundtrips_typed_annotation() -> None:
    payload = SQLCapabilityExtractor().extract(
        "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
        example_id="ex1",
        dataset_source="mock",
        database_identifier="db1",
        schema={"tables": {"orders": {"columns": {"customer_id": {}}}}},
    ).model_dump(mode="json")

    loaded = CapabilityAnnotation.model_validate(payload)

    assert loaded.example_id == "ex1"
    assert loaded.dataset_source == "mock"
    assert "AGGREGATION" in loaded.required_capabilities
    assert loaded.partial_supervision.group_by_columns == ["customer_id"]


# Source: tests/test_capability_dataset_reporting.py
from capabilities import CapabilityDatasetReporter, SQLCapabilityExtractor
from capabilities.evaluation import CapabilityEvaluator


def _row(sql: str, split: str, unsupported_reason: str | None = None) -> dict:
    annotation = SQLCapabilityExtractor().extract(
        sql,
        example_id=f"{split}:{sql[:8]}",
        dataset_source="mock",
        database_identifier="db",
        full_query_ir_supported=unsupported_reason is None,
        unsupported_reason=unsupported_reason,
    ).model_dump(mode="json")
    return {
        "dataset_name": "mock",
        "split": split,
        "unsupported_reason": unsupported_reason,
        "required_capabilities": annotation["required_capabilities"],
        "partial_supervision": annotation["partial_supervision"],
        "task_masks": annotation["task_masks"],
        "capability_annotation": annotation,
    }


def test_capability_dataset_reporting_summarizes_distribution_and_retention() -> None:
    rows = [
        _row("SELECT id FROM users", "train"),
        _row("SELECT RANK() OVER (ORDER BY amount DESC) FROM orders", "unsupported", "window_function"),
        _row("SELECT id FROM a UNION SELECT id FROM b", "unsupported", "set_operation"),
    ]
    reporter = CapabilityDatasetReporter(rare_threshold=2)

    report = reporter.build_report(rows)
    retention = reporter.build_retention_report(rows)

    assert report["summary"]["total_examples"] == 3
    assert report["summary"]["partial_supervision_only_count"] == 2
    assert report["capability_frequency"]["WINDOW_RANK"] == 1
    assert report["set_operation_distribution"]["UNION"] == 1
    assert "zero_coverage_capabilities" in report
    assert report["summary"]["partial_supervision_extraction_coverage"] == 1.0
    assert retention["summary"]["unsupported_examples"] == 2
    assert retention["summary"]["retained_for_auxiliary_supervision"] == 2


def test_capability_evaluator_computes_multilabel_metrics() -> None:
    report = CapabilityEvaluator().evaluate(
        gold_labels=[["SIMPLE_SELECT", "FILTER"], ["SIMPLE_SELECT", "AGGREGATION"]],
        predicted_scores=[
            {"SIMPLE_SELECT": 0.9, "FILTER": 0.8, "AGGREGATION": 0.1},
            {"SIMPLE_SELECT": 0.9, "FILTER": 0.2, "AGGREGATION": 0.7},
        ],
        thresholds={"SIMPLE_SELECT": 0.5, "FILTER": 0.5, "AGGREGATION": 0.5},
    )

    assert report["micro_f1"] == 1.0
    assert report["exact_multilabel_match"] == 1.0
    assert report["per_capability"]["FILTER"]["average_precision"] == 1.0


# Source: tests/test_capability_or_filter_consistency.py
from capabilities.sql_capability_extractor import SQLCapabilityExtractor
from ir.query_ir_v2_models import FromItem, QueryNode
from ir.query_ir_v2_validation import QueryIRV2Validator
from ir.sql_to_query_ir_v2 import SQLToQueryIRV2Converter
from tests.query_ir_v2_boolean_helpers import eq, or_tree


def test_phase1_extractor_continues_labeling_or_filter() -> None:
    annotation = SQLCapabilityExtractor().extract("SELECT id FROM customers WHERE region = 'US' OR region = 'CA'")

    assert "FILTER" in annotation.required_capabilities
    assert "MULTIPLE_FILTERS" in annotation.required_capabilities
    assert "OR_FILTER" in annotation.required_capabilities


def test_sql_to_v2_converter_produces_or_tree_matching_capability_label() -> None:
    query = SQLToQueryIRV2Converter().convert("SELECT id FROM customers WHERE region = 'US' OR region = 'CA'")

    result = QueryIRV2Validator().validate(query)

    assert result.is_valid
    assert "OR_FILTER" in query.capability_metadata.source_capability_labels


def test_validator_rejects_or_tree_without_or_filter_label() -> None:
    query = QueryNode(
        from_item=FromItem(table="customers"),
        where=or_tree(eq("region", "US"), eq("region", "CA")),
    )

    result = QueryIRV2Validator().validate(query)

    assert not result.is_valid
    assert any(issue.issue_type == "or_filter_capability_mismatch" for issue in result.issues)


# Source: tests/test_capability_taxonomy.py
from capabilities import Capability, SafetyLabel
from capabilities.taxonomy import SUPPORTED_QUERYIR_V1_CAPABILITIES, capability_names


def test_capability_taxonomy_is_multilabel_and_separates_safety() -> None:
    capabilities = {
        Capability.AGGREGATION,
        Capability.GROUP_BY,
        Capability.WINDOW_RANK,
        Capability.MULTI_HOP_JOIN,
        Capability.ORDER_BY,
    }
    safety_labels = {SafetyLabel.MUTATION_DELETE}

    assert len(capabilities) == 5
    assert SafetyLabel.MUTATION_DELETE not in capabilities
    assert Capability.WINDOW_RANK not in SUPPORTED_QUERYIR_V1_CAPABILITIES
    assert capability_names(capabilities) == [
        "AGGREGATION",
        "GROUP_BY",
        "MULTI_HOP_JOIN",
        "ORDER_BY",
        "WINDOW_RANK",
    ]


# Source: tests/test_capability_training_batch.py
from capabilities import ALL_CAPABILITIES, ALL_SAFETY_LABELS
from neural_ir.ir_dataset import capability_label_vector, collate_ir_batch, safety_label_vector, task_mask_vector


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
        "safety_labels": safety_label_vector(row),
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


def test_safety_training_batch_adds_separate_multilabel_targets() -> None:
    row = {
        "safety_labels": ["MUTATION_DELETE", "UNSAFE_REQUEST"],
        "task_masks": {"safety": 1, "capability": 0, "full_query_ir": 0},
    }
    batch = collate_ir_batch([_item(row)])
    safety_index = {label.value: index for index, label in enumerate(ALL_SAFETY_LABELS)}

    assert batch["safety_labels"].shape == (1, len(ALL_SAFETY_LABELS))
    assert batch["safety_labels"][0, safety_index["MUTATION_DELETE"]].item() == 1.0
    assert batch["safety_labels"][0, safety_index["UNSAFE_REQUEST"]].item() == 1.0
    assert batch["task_masks"]["safety"].item() == 1.0


# Source: tests/test_sql_capability_extractor.py
import pytest

from capabilities import SQLCapabilityExtractor


def _caps(sql: str) -> set[str]:
    return set(SQLCapabilityExtractor().extract(sql).required_capabilities)


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id", {"AGGREGATION", "GROUP_BY"}),
        ("SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id HAVING SUM(amount) > 10", {"HAVING"}),
        ("SELECT CASE WHEN amount > 10 THEN 'big' ELSE 'small' END FROM orders", {"CASE_EXPRESSION"}),
        ("SELECT (SELECT MAX(amount) FROM orders) AS max_amount", {"SCALAR_SUBQUERY", "AGGREGATION"}),
        ("SELECT id FROM customers WHERE id IN (SELECT customer_id FROM orders)", {"IN_SUBQUERY"}),
        ("SELECT id FROM customers WHERE EXISTS (SELECT 1 FROM orders)", {"EXISTS_SUBQUERY"}),
        ("SELECT t.customer_id FROM (SELECT customer_id FROM orders) AS t", {"DERIVED_TABLE"}),
        (
            "SELECT c.id FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)",
            {"EXISTS_SUBQUERY", "CORRELATED_SUBQUERY"},
        ),
        ("SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) FROM orders", {"WINDOW_ROW_NUMBER"}),
        ("SELECT RANK() OVER (ORDER BY amount DESC) FROM orders", {"WINDOW_RANK"}),
        ("SELECT LAG(amount) OVER (ORDER BY order_date) FROM orders", {"WINDOW_LAG"}),
        ("SELECT SUM(amount) OVER (PARTITION BY customer_id) FROM orders", {"WINDOW_AGGREGATE"}),
        ("SELECT id FROM a UNION ALL SELECT id FROM b", {"UNION_ALL"}),
        ("SELECT id FROM a UNION SELECT id FROM b", {"UNION"}),
        ("SELECT id FROM a INTERSECT SELECT id FROM b", {"INTERSECT"}),
        ("SELECT id FROM a EXCEPT SELECT id FROM b", {"EXCEPT"}),
    ],
)
def test_sql_capability_extractor_detects_required_capabilities(sql: str, expected: set[str]) -> None:
    assert expected.issubset(_caps(sql))


def test_sql_capability_extractor_detects_safety_labels() -> None:
    extractor = SQLCapabilityExtractor()
    assert extractor.extract("INSERT INTO orders(id) VALUES (1)").safety_labels == ["MUTATION_INSERT"]
    assert extractor.extract("UPDATE orders SET amount = 1").safety_labels == ["MUTATION_UPDATE"]
    assert extractor.extract("DELETE FROM orders WHERE id = 1").safety_labels == ["MUTATION_DELETE"]
    assert extractor.extract("CREATE TABLE t(id INT)").safety_labels == ["DDL_CREATE"]


def test_multiple_simultaneous_capabilities_and_policy_are_separate() -> None:
    annotation = SQLCapabilityExtractor().extract(
        "SELECT c.region, SUM(o.amount) AS revenue "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "WHERE c.region = 'west' OR c.region = 'east' "
        "GROUP BY c.region ORDER BY revenue DESC LIMIT 3"
    )

    assert {
        "AGGREGATION",
        "GROUP_BY",
        "ONE_HOP_JOIN",
        "OR_FILTER",
        "ORDER_BY",
        "LIMIT",
    }.issubset(set(annotation.required_capabilities))
    assert annotation.understood is True
    assert annotation.currently_supported is False
    assert annotation.unsupported_required_capabilities == ["OR_FILTER"]


# Source: tests/test_training_inference_capability_parity.py
from capabilities import SQLCapabilityExtractor


def test_training_and_inference_capability_extraction_are_deterministic() -> None:
    sql = (
        "SELECT c.region, SUM(o.amount) AS revenue "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "WHERE c.region = 'west' "
        "GROUP BY c.region ORDER BY revenue DESC LIMIT 3"
    )
    extractor = SQLCapabilityExtractor()
    training_annotation = extractor.extract(
        sql,
        example_id="train-example",
        dataset_source="training",
        database_identifier="sales",
    )
    inference_annotation = extractor.extract(
        sql,
        example_id="runtime-example",
        dataset_source="runtime",
        database_identifier="sales",
    )

    assert training_annotation.required_capabilities == inference_annotation.required_capabilities
    assert training_annotation.partial_supervision.referenced_tables == inference_annotation.partial_supervision.referenced_tables
    assert training_annotation.partial_supervision.filter_columns == inference_annotation.partial_supervision.filter_columns
    assert training_annotation.partial_supervision.join_edges == inference_annotation.partial_supervision.join_edges


# Source: tests/test_unsupported_example_task_masks.py
from capabilities import SQLCapabilityExtractor
from capabilities.contracts import annotation_to_unsupported_example


def test_unsupported_example_uses_auxiliary_masks_without_full_ir_loss() -> None:
    sql = "SELECT RANK() OVER (ORDER BY amount DESC) AS rnk FROM orders"
    extractor = SQLCapabilityExtractor()
    annotation = extractor.extract(sql, unsupported_reason="window_function")
    annotation = extractor.with_conversion_result(annotation, {"success": False, "unsupported_reason": "window_function"})
    example = annotation_to_unsupported_example(annotation, "window_function")

    assert "WINDOW_RANK" in example.capabilities
    assert example.task_masks.capability == 1
    assert example.task_masks.table == 1
    assert example.task_masks.window == 1
    assert example.task_masks.full_query_ir == 0
    assert example.partial_supervision.full_query_ir_supported is False
