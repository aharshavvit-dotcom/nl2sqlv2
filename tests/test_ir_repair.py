from __future__ import annotations

from neural_ir.ir_repair import OptionAIRRepairer


def test_ir_repair_missing_limit_and_count_metric() -> None:
    result = OptionAIRRepairer().repair(_base_ir(intent="count_records", limit=0, metrics=[]), _schema(), "How many orders?")

    repaired = result["query_ir"]
    assert "added_default_limit_100" in result["repairs_applied"]
    assert "added_count_star_metric" in result["repairs_applied"]
    assert repaired["limit"] == 100
    assert repaired["metrics"][0]["expression"] == "*"


def test_ir_repair_product_revenue_to_item_level_expression() -> None:
    query_ir = _base_ir(
        intent="top_n_metric_by_dimension",
        metrics=[{"name": "revenue", "aggregation": "SUM", "table": "orders", "column": "amount", "expression": "orders.amount", "alias": "revenue"}],
        dimensions=[{"name": "product", "table": "products", "column": "product_name", "expression": "products.product_name", "alias": "product_name"}],
        required_tables=["orders", "products"],
    )

    result = OptionAIRRepairer().repair(query_ir, _schema(), "Top 5 products by revenue")

    assert "corrected_product_revenue_to_item_level_expression" in result["repairs_applied"]
    assert result["query_ir"]["metrics"][0]["expression"] == "order_items.quantity * order_items.price"


def test_ir_repair_date_grain_and_missing_join() -> None:
    query_ir = _base_ir(
        intent="trend_by_date",
        metrics=[{"name": "sales", "aggregation": "SUM", "table": "orders", "column": "amount", "expression": "orders.amount", "alias": "sales"}],
        dimensions=[{"name": "customer", "table": "customers", "column": "customer_name", "expression": "customers.customer_name", "alias": "customer_name"}],
        required_tables=["orders", "customers"],
        date_filters=[],
        joins=[],
    )

    result = OptionAIRRepairer().repair(query_ir, _schema(), "Show sales by month")

    assert "inferred_date_grain_month" in result["repairs_applied"]
    assert "added_missing_join_plan" in result["repairs_applied"]
    assert result["query_ir"]["joins"]


def _base_ir(
    intent: str,
    limit: int = 100,
    metrics: list[dict] | None = None,
    dimensions: list[dict] | None = None,
    required_tables: list[str] | None = None,
    date_filters: list[dict] | None = None,
    joins: list[dict] | None = None,
) -> dict:
    return {
        "query_ir_id": "test",
        "question": "q",
        "normalized_question": "q",
        "intent": intent,
        "template_id": intent,
        "dialect": "sqlite",
        "base_table": "orders",
        "required_tables": required_tables or ["orders"],
        "metrics": metrics if metrics is not None else [{"name": "sales", "aggregation": "SUM", "table": "orders", "column": "amount", "expression": "orders.amount", "alias": "sales"}],
        "dimensions": dimensions or [],
        "filters": [],
        "date_filters": date_filters if date_filters is not None else [],
        "joins": joins if joins is not None else [],
        "group_by": [],
        "order_by": [],
        "limit": limit,
        "select_mode": "aggregate",
        "warnings": [],
        "metadata": {},
    }


def _schema() -> dict:
    return {
        "tables": {
            "orders": {"columns": {"order_id": {"type": "INTEGER"}, "customer_id": {"type": "INTEGER"}, "amount": {"type": "FLOAT"}, "order_date": {"type": "DATE"}}},
            "customers": {"columns": {"customer_id": {"type": "INTEGER"}, "customer_name": {"type": "TEXT"}}},
            "order_items": {"columns": {"order_id": {"type": "INTEGER"}, "product_id": {"type": "INTEGER"}, "quantity": {"type": "INTEGER"}, "price": {"type": "FLOAT"}}},
            "products": {"columns": {"product_id": {"type": "INTEGER"}, "product_name": {"type": "TEXT"}}},
        },
        "foreign_keys": [
            {"from_table": "orders", "from_column": "customer_id", "to_table": "customers", "to_column": "customer_id"},
            {"from_table": "order_items", "from_column": "order_id", "to_table": "orders", "to_column": "order_id"},
            {"from_table": "order_items", "from_column": "product_id", "to_table": "products", "to_column": "product_id"},
        ],
    }
