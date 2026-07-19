"""
Purpose: Verifies ir integration behaviour consolidated from fragmented test files.
Required because: Renderer output and SQLite execution equivalence should be exercised as integration behaviour.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# Source: tests/test_query_ir_v2_boolean_precedence.py
from ir.query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer
from ir.sql_to_query_ir_v2 import SQLToQueryIRV2Converter
from tests.query_ir_v2_boolean_helpers import execute_rows, sample_connection


def test_parentheses_change_predicate_tree_and_execution_results() -> None:
    converter = SQLToQueryIRV2Converter()
    without_parentheses = converter.convert(
        "SELECT id FROM customers WHERE region = 'US' OR region = 'CA' AND status = 'ACTIVE'"
    )
    with_parentheses = converter.convert(
        "SELECT id FROM customers WHERE (region = 'US' OR region = 'CA') AND status = 'ACTIVE'"
    )

    assert without_parentheses.where.model_dump() != with_parentheses.where.model_dump()  # type: ignore[union-attr]

    renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
    conn = sample_connection()
    left_rows = execute_rows(conn, renderer.render(without_parentheses))
    right_rows = execute_rows(conn, renderer.render(with_parentheses))

    assert left_rows != right_rows


def test_renderer_parenthesizes_and_under_or_for_exact_tree_preservation() -> None:
    query = SQLToQueryIRV2Converter().convert(
        "SELECT id FROM customers WHERE region = 'US' OR (region = 'CA' AND status = 'ACTIVE')"
    )

    sql = QueryIRV2NativeRenderer(enable_or_rendering=True).render(query)

    assert " OR (" in sql
    assert " AND " in sql


# Source: tests/test_query_ir_v2_boolean_execution_equivalence.py
from ir.query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer
from ir.sql_to_query_ir_v2 import SQLToQueryIRV2Converter
from tests.query_ir_v2_boolean_helpers import execute_rows, load_boolean_eval_cases, sample_connection


def test_curated_boolean_eval_cases_parse_render_and_execute_equivalently() -> None:
    converter = SQLToQueryIRV2Converter()
    renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
    conn = sample_connection()

    for case in load_boolean_eval_cases():
        query = converter.convert(case["sql"])
        rendered = renderer.render(query)
        reparsed = converter.convert(rendered)

        assert reparsed.where.model_dump() == query.where.model_dump(), case["id"]  # type: ignore[union-attr]
        assert execute_rows(conn, rendered) == execute_rows(conn, case["sql"]), case["id"]
