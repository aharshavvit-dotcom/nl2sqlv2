from __future__ import annotations

from capabilities import SQLCapabilityExtractor
from ir.sql_to_ir_converter import SQLToIRConverter


def test_partial_supervision_retains_labels_for_unsupported_window_query() -> None:
    sql = (
        "SELECT customer_id, ROW_NUMBER() OVER "
        "(PARTITION BY region ORDER BY amount DESC) AS rn FROM orders"
    )
    result = SQLToIRConverter().convert("rank orders", sql, schema=None)
    annotation = SQLCapabilityExtractor().with_conversion_result(
        SQLCapabilityExtractor().extract(sql),
        result,
    )
    partial = annotation.partial_supervision

    assert result["success"] is False
    assert "WINDOW_ROW_NUMBER" in annotation.required_capabilities
    assert partial.full_query_ir_supported is False
    assert partial.unsupported_reason == "window_function"
    assert partial.referenced_tables == ["orders"]
    assert "customer_id" in partial.selected_columns
    assert partial.window_functions[0].function == "ROW_NUMBER"
    assert annotation.task_masks.window == 1
    assert annotation.task_masks.full_query_ir == 0


def test_partial_supervision_captures_correlated_subquery_details() -> None:
    sql = (
        "SELECT c.id FROM customers c "
        "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)"
    )
    partial = SQLCapabilityExtractor().extract(sql).partial_supervision

    assert "CORRELATED_SUBQUERY" in partial.subquery_types
    assert partial.subquery_depth >= 1
    assert partial.correlated_subqueries
    assert "c.id" in partial.correlated_subqueries[0].correlated_columns
    assert "EQ" in partial.correlated_subqueries[0].correlation_operators


def test_partial_supervision_captures_set_operation_branches() -> None:
    partial = SQLCapabilityExtractor().extract("SELECT id FROM a UNION ALL SELECT id FROM b").partial_supervision

    assert partial.set_operation == "UNION_ALL"
    assert len(partial.set_operation_branches) == 2
    assert partial.set_operation_branches[0].required_capabilities
