from __future__ import annotations

import sqlglot

from ir.sql_to_ir_rules import (
    date_grain_from_sql,
    extract_aggregations,
    extract_group_by,
    extract_joins,
    extract_limit,
    extract_order_by,
    extract_tables,
    extract_where_filters,
)


SQL = """
SELECT customers.customer_name AS customer, SUM(orders.amount) AS revenue
FROM orders
JOIN customers ON orders.customer_id = customers.customer_id
WHERE orders.status = 'completed'
GROUP BY customers.customer_name
ORDER BY revenue DESC
LIMIT 5
"""


def test_sql_to_ir_rules_extract_core_features() -> None:
    ast = sqlglot.parse_one(SQL, read="sqlite")

    assert extract_tables(ast) == ["orders", "customers"]
    assert extract_joins(ast)[0]["condition"] == "orders.customer_id = customers.customer_id"
    assert extract_aggregations(ast)[0]["function"] == "SUM"
    assert extract_group_by(ast) == ["customers.customer_name"]
    assert extract_order_by(ast)[0]["direction"] == "DESC"
    assert extract_limit(ast) == 5
    assert extract_where_filters(ast)[0]["left"]["expression"] == "orders.status"


def test_sql_to_ir_rules_detect_date_grain_expression() -> None:
    grain = date_grain_from_sql("strftime('%Y-%m', orders.order_date)")

    assert grain == {
        "date_grain": "month",
        "date_table": "orders",
        "date_column": "order_date",
        "date_expression": "orders.order_date",
    }

