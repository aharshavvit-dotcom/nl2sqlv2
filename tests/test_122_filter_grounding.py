from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from inference.grounding.filter_value_contract import ExtractedLiteral
from inference.grounding.filter_value_extractor import FilterValueExtractionContract
from inference.grounding.filter_grounding_service import FilterGroundingService
from inference.grounding.schema_value_index import SchemaValueIndex
from inference.runtime_schema_context import RuntimeSchemaContext


def test_join_graph_relevance_resolves_conflict():
    val_idx = MagicMock(spec=SchemaValueIndex)
    val_idx.lookup_value.return_value = [
        {"table": "customers", "column": "customers.region", "score": 0.94, "signals": {"exact_value_match": 1.0}},
        {"table": "suppliers", "column": "suppliers.region", "score": 0.94, "signals": {"exact_value_match": 1.0}},
    ]

    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.foreign_keys = [{"child_table": "orders", "parent_table": "customers"}]
    ctx.get_columns.return_value = ["customers.region", "suppliers.region"]
    ctx.column_info.return_value = {"is_sensitive": False}

    service = FilterGroundingService(val_idx, ctx)

    contract = FilterValueExtractionContract(
        raw_question="show orders from customers in the west",
        extracted_literals=[
            ExtractedLiteral(
                literal_id="lit_0",
                raw_text="west",
                normalized_value="west",
                value_type="string",
                span_start=34,
                span_end=38,
                extraction_method="quoted_string",
                extraction_confidence=0.9,
            )
        ],
    )

    res = service.ground_filters("show orders from customers in the west", contract, entity_table="orders")

    assert res[0].selected_candidate is not None
    assert res[0].selected_candidate.table_name == "customers"
    assert res[0].selected_candidate.column_name == "region"


def test_ambiguity_requires_clarification():
    val_idx = MagicMock(spec=SchemaValueIndex)
    val_idx.lookup_value.return_value = [
        {"table": "employees", "column": "employees.account_status", "score": 0.85, "signals": {"exact_value_match": 1.0}},
        {"table": "employees", "column": "employees.employment_status", "score": 0.84, "signals": {"exact_value_match": 1.0}},
    ]

    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.foreign_keys = []
    ctx.get_columns.return_value = ["employees.account_status", "employees.employment_status"]
    ctx.column_info.return_value = {"is_sensitive": False}

    service = FilterGroundingService(val_idx, ctx)

    contract = FilterValueExtractionContract(
        raw_question="show active employees",
        extracted_literals=[
            ExtractedLiteral(
                literal_id="lit_0",
                raw_text="active",
                normalized_value="active",
                value_type="string",
                span_start=5,
                span_end=11,
                extraction_method="quoted_string",
                extraction_confidence=0.9,
            )
        ],
    )

    res = service.ground_filters("show active employees", contract, entity_table="employees")
    assert res[0].requires_clarification is True
    assert "Does 'active' refer to" in res[0].clarification_question
