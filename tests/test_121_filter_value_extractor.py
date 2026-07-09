from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
import pytest
from inference.grounding.filter_value_contract import QueryTimeContext
from inference.grounding.filter_value_extractor import FilterValueExtractor
from inference.grounding.schema_value_index import SchemaValueIndex


def test_negative_number_extraction():
    idx = MagicMock(spec=SchemaValueIndex)
    extractor = FilterValueExtractor(idx)

    contract = extractor.extract_literals("temperature was below -15 degrees")
    literals = {l.normalized_value for l in contract.extracted_literals}
    assert -15.0 in literals or -15 in literals


def test_relative_date_with_time_context():
    idx = MagicMock(spec=SchemaValueIndex)
    extractor = FilterValueExtractor(idx)

    time_ctx = QueryTimeContext(current_datetime=datetime(2026, 7, 9))
    contract = extractor.extract_literals("orders placed yesterday", time_context=time_ctx)
    yesterday_lit = [l for l in contract.extracted_literals if l.raw_text == "yesterday"]
    assert len(yesterday_lit) == 1
    assert yesterday_lit[0].normalized_value == "2026-07-08"


def test_list_extraction():
    idx = MagicMock(spec=SchemaValueIndex)
    extractor = FilterValueExtractor(idx)

    contract = extractor.extract_literals("customers in India, Japan, or Singapore")
    list_lits = [l for l in contract.extracted_literals if l.value_type == "list"]
    assert len(list_lits) == 1
    assert list_lits[0].normalized_value == ["India", "Japan", "Singapore"]
