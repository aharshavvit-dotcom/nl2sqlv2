from __future__ import annotations

from neural_ir.candidate_builder import SchemaCandidateBuilder
from neural_ir.schema_linker import SchemaLinker


def test_schema_linker_links_business_terms_to_columns() -> None:
    candidates = SchemaCandidateBuilder().build_candidates(_schema())
    linker = SchemaLinker()

    assert _top_column(linker.link("sales", candidates)) == "orders.amount"
    assert _top_column(linker.link("customer", candidates)) == "customers.customer_name"
    assert _top_column(linker.link("product", candidates)) == "products.product_name"
    assert _top_column(linker.link("status", candidates)) == "orders.status"
    assert _top_column(linker.link("month", candidates)) == "orders.order_date"


def _top_column(result: dict) -> str:
    return result["top_columns"][0]["display"]


def _schema() -> dict:
    return {
        "tables": {
            "orders": {"columns": {"order_id": {"type": "INTEGER"}, "amount": {"type": "FLOAT"}, "order_date": {"type": "DATE"}, "status": {"type": "TEXT"}}},
            "customers": {"columns": {"customer_id": {"type": "INTEGER"}, "customer_name": {"type": "TEXT"}, "region": {"type": "TEXT"}}},
            "products": {"columns": {"product_id": {"type": "INTEGER"}, "product_name": {"type": "TEXT"}, "category": {"type": "TEXT"}}},
        }
    }

