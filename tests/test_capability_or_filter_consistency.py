from __future__ import annotations

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
