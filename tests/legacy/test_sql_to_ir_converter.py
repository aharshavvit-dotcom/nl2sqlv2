"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ir.sql_to_ir_converter import SQLToIRConverter
from nl2sql_v1.schema import read_sqlite_schema
from scripts.create_sample_db import build_database


@pytest.fixture()
def sample_schema(tmp_path: Path):
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    return read_sqlite_schema(db_path)


@pytest.mark.parametrize(
    ("sql", "intent"),
    [
        ("SELECT SUM(orders.amount) AS revenue FROM orders LIMIT 100", "metric_summary"),
        (
            "SELECT customers.region AS region, SUM(orders.amount) AS revenue FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id GROUP BY customers.region ORDER BY revenue DESC LIMIT 100",
            "metric_by_dimension",
        ),
        (
            "SELECT customers.customer_name AS customer, SUM(orders.amount) AS revenue FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id GROUP BY customers.customer_name ORDER BY revenue DESC LIMIT 5",
            "top_n_metric_by_dimension",
        ),
        (
            "SELECT customers.customer_name AS customer, SUM(orders.amount) AS revenue FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id GROUP BY customers.customer_name ORDER BY revenue ASC LIMIT 5",
            "bottom_n_metric_by_dimension",
        ),
        ("SELECT COUNT(*) AS record_count FROM orders LIMIT 100", "count_records"),
        (
            "SELECT orders.status AS status, COUNT(*) AS record_count FROM orders "
            "GROUP BY orders.status ORDER BY record_count DESC LIMIT 100",
            "count_by_dimension",
        ),
        (
            "SELECT strftime('%Y-%m', orders.order_date) AS period, SUM(orders.amount) AS revenue "
            "FROM orders GROUP BY strftime('%Y-%m', orders.order_date) ORDER BY period ASC LIMIT 100",
            "trend_by_date",
        ),
        (
            "SELECT orders.order_id, orders.status, orders.order_date FROM orders "
            "WHERE orders.status = 'completed' LIMIT 100",
            "simple_filter",
        ),
    ],
)
def test_sql_to_ir_converter_supported_patterns(sample_schema, sql: str, intent: str) -> None:
    result = SQLToIRConverter().convert("question", sql, sample_schema)

    assert result["success"], result
    assert result["query_ir"]["intent"] == intent
    assert result["ir_validation"]["is_valid"]
    assert result["sql_validation"]["is_valid"]
    assert result["roundtrip_validation"]["is_valid"]


def test_sql_to_ir_converter_product_revenue_expression(sample_schema) -> None:
    sql = (
        "SELECT products.product_name AS product, SUM(order_items.quantity * order_items.price) AS revenue "
        "FROM order_items JOIN products ON order_items.product_id = products.product_id "
        "GROUP BY products.product_name ORDER BY revenue DESC LIMIT 5"
    )

    result = SQLToIRConverter().convert("Product revenue", sql, sample_schema)

    assert result["success"], result
    query_ir = result["query_ir"]
    assert query_ir["base_table"] == "order_items"
    assert query_ir["metrics"][0]["expression"] == "order_items.quantity * order_items.price"


def test_sql_to_ir_converter_accepts_bare_metric_column(sample_schema) -> None:
    result = SQLToIRConverter().convert(
        "Total order amount",
        "SELECT SUM(amount) FROM orders WHERE status = 'completed'",
        sample_schema,
    )

    assert result["success"], result
    assert result["query_ir"]["metrics"][0]["expression"] == "orders.amount"


def test_sql_to_ir_converter_accepts_table_aliases_in_join(sample_schema) -> None:
    sql = (
        "SELECT T2.region FROM orders AS T1 INNER JOIN customers AS T2 "
        "ON T1.customer_id = T2.customer_id WHERE T1.status = 'completed'"
    )

    result = SQLToIRConverter().convert("Regions with completed orders", sql, sample_schema)

    assert result["success"], result
    assert result["roundtrip_validation"]["checks"]["joins_compatible"]


def test_sql_to_ir_converter_preserves_count_column(sample_schema) -> None:
    result = SQLToIRConverter().convert(
        "Count orders with an amount",
        "SELECT COUNT(amount) FROM orders",
        sample_schema,
    )

    assert result["success"], result
    assert "COUNT(orders.amount)" in result["roundtrip_sql"]
    assert "COUNT(*)" not in result["roundtrip_sql"]


def test_sql_to_ir_converter_date_range_filter(sample_schema) -> None:
    sql = (
        "SELECT SUM(orders.amount) AS revenue FROM orders "
        "WHERE orders.order_date >= '2026-05-01' AND orders.order_date < '2026-06-01' LIMIT 100"
    )

    result = SQLToIRConverter().convert("Sales last month", sql, sample_schema)

    assert result["success"], result
    date_filter = result["query_ir"]["date_filters"][0]
    assert date_filter["start_date"] == "2026-05-01"
    assert date_filter["end_date"] == "2026-06-01"


def test_sql_to_ir_converter_unsupported_nested_and_union(sample_schema) -> None:
    nested = SQLToIRConverter().convert(
        "nested",
        "SELECT orders.order_id FROM orders WHERE orders.amount > (SELECT AVG(orders.amount) FROM orders)",
        sample_schema,
    )
    union = SQLToIRConverter().convert(
        "union",
        "SELECT orders.order_id FROM orders UNION SELECT orders.order_id FROM orders",
        sample_schema,
    )

    assert nested["success"] is False
    assert nested["unsupported_reason"] == "nested_query"
    assert union["success"] is False
    assert union["unsupported_reason"] == "set_operation"


def test_sql_to_ir_converter_unsupported_non_equality_join(sample_schema) -> None:
    result = SQLToIRConverter().convert(
        "bad join",
        "SELECT orders.order_id FROM orders JOIN customers ON orders.amount > customers.customer_id LIMIT 100",
        sample_schema,
    )

    assert result["success"] is False
    assert result["unsupported_reason"] == "unsupported_join"
