from __future__ import annotations

from neural_ir.candidate_builder import SchemaCandidateBuilder
from neural_ir.hard_negative_builder import HardNegativeBuilder


def test_hard_negative_builder_creates_metric_dimension_and_product_rows() -> None:
    example = _example()
    candidates = SchemaCandidateBuilder().build_candidates(_schema(), example["question"])

    rows = HardNegativeBuilder().build_for_example(example, candidates)
    negative_types = {row["negative_type"] for row in rows}

    assert "wrong_metric_column" in negative_types
    assert "wrong_dimension_column" in negative_types
    assert "product_revenue_wrong_grain" in negative_types
    assert all(row["metadata"]["reason"] for row in rows)


def _example() -> dict:
    return {
        "example_id": "x1",
        "question": "Top 5 products by revenue",
        "query_ir": {
            "intent": "top_n_metric_by_dimension",
            "template_id": "top_n_metric_by_dimension",
            "base_table": "order_items",
            "metrics": [{"name": "revenue", "aggregation": "SUM", "table": "order_items", "column": None, "expression": "order_items.quantity * order_items.price", "alias": "revenue"}],
            "dimensions": [{"name": "product", "table": "products", "column": "product_name", "expression": "products.product_name", "alias": "product_name"}],
            "filters": [],
            "date_filters": [],
            "order_by": [{"direction": "DESC"}],
            "limit": 5,
        },
    }


def _schema() -> dict:
    return {
        "tables": {
            "orders": {"columns": {"amount": {"type": "FLOAT"}}},
            "order_items": {"columns": {"quantity": {"type": "INTEGER"}, "price": {"type": "FLOAT"}, "product_id": {"type": "INTEGER"}}},
            "products": {"columns": {"product_id": {"type": "INTEGER"}, "product_name": {"type": "TEXT"}, "category": {"type": "TEXT"}}},
        }
    }
