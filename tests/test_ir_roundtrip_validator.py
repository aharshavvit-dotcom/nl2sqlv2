from __future__ import annotations

from pathlib import Path

from ir.ir_roundtrip_validator import IRRoundtripValidator
from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.query_ir_models import QueryIR
from ir.sql_to_ir_converter import SQLToIRConverter
from nl2sql_v1.schema import read_sqlite_schema
from scripts.create_sample_db import build_database


def test_ir_roundtrip_validator_accepts_compatible_rendered_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    source_sql = (
        "SELECT customers.customer_name AS customer, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.customer_id "
        "GROUP BY customers.customer_name ORDER BY revenue DESC LIMIT 5"
    )
    result = SQLToIRConverter().convert("Top customers", source_sql, schema)
    query_ir = QueryIR(**result["query_ir"])
    rendered = IRToSQLRenderer().render(query_ir)

    validation = IRRoundtripValidator().validate_roundtrip(source_sql, query_ir, rendered, schema=schema)

    assert validation["is_valid"], validation
    assert validation["checks"]["tables_compatible"]
    assert validation["checks"]["metrics_compatible"]
    assert validation["checks"]["dimensions_compatible"]
    assert validation["checks"]["filters_compatible"]

