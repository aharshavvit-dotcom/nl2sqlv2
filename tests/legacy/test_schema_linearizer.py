from __future__ import annotations

from neural_ir.schema_linearizer import SchemaLinearizer


def test_schema_linearizer_outputs_schema_text_and_typed_items() -> None:
    schema = {
        "tables": {
            "customers": {
                "columns": {
                    "customer_id": {"type": "INTEGER"},
                    "customer_name": {"type": "TEXT"},
                    "region": {"type": "TEXT"},
                }
            },
            "orders": {
                "columns": {
                    "order_id": {"type": "INTEGER"},
                    "order_date": {"type": "TEXT"},
                    "amount": {"type": "FLOAT"},
                    "status": {"type": "TEXT"},
                }
            },
        }
    }

    linearizer = SchemaLinearizer()
    text = linearizer.linearize(schema)
    items = linearizer.extract_schema_items(schema)

    assert text == "tables: customers(customer_id, customer_name, region); orders(order_id, order_date, amount, status)"
    assert items["columns"][0] == {"index": 0, "table": "customers", "column": "customer_id", "type": "id"}
    assert any(item["column"] == "amount" for item in items["numeric_columns"])
    assert any(item["column"] == "order_date" for item in items["date_columns"])
    assert any(item["column"] == "customer_name" for item in items["text_columns"])
    assert any(item["column"] == "order_id" for item in items["id_columns"])

