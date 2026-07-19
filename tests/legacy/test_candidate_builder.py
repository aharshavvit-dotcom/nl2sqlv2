"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from neural_ir.candidate_builder import SchemaCandidateBuilder


def test_candidate_builder_roles_columns_by_type() -> None:
    candidates = SchemaCandidateBuilder().build_candidates(_schema())

    metric_columns = {item["display"] for item in candidates["metric_candidates"]}
    dimension_columns = {item["display"] for item in candidates["dimension_candidates"]}
    date_columns = {item["display"] for item in candidates["date_candidates"]}
    filter_columns = {item["display"] for item in candidates["filter_candidates"]}

    assert "orders.amount" in metric_columns
    assert "customers.customer_name" in dimension_columns
    assert "orders.order_date" in date_columns
    assert {"orders.status", "customers.region", "products.category"}.issubset(filter_columns)


def _schema() -> dict:
    return {
        "tables": {
            "orders": {"columns": {"order_id": {"type": "INTEGER"}, "amount": {"type": "FLOAT"}, "order_date": {"type": "DATE"}, "status": {"type": "TEXT"}}},
            "customers": {"columns": {"customer_id": {"type": "INTEGER"}, "customer_name": {"type": "TEXT"}, "region": {"type": "TEXT"}}},
            "products": {"columns": {"product_id": {"type": "INTEGER"}, "product_name": {"type": "TEXT"}, "category": {"type": "TEXT"}}},
        }
    }

