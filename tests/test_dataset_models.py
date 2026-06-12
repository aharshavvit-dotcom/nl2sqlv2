from __future__ import annotations

from datasets.models import DatabaseSchema, Text2SQLExample


def test_text2sql_example_validates_defaults() -> None:
    example = Text2SQLExample(
        example_id="ex1",
        dataset_name="mock",
        db_id="db",
        question="How many orders?",
        sql="SELECT COUNT(*) FROM orders",
        split="train",
    )

    assert example.tables == []
    assert example.extracted_slots == {}
    assert example.to_dict()["example_id"] == "ex1"


def test_database_schema_serializes() -> None:
    schema = DatabaseSchema(db_id="db", dataset_name="mock", tables={"orders": {"columns": ["id"]}})
    assert schema.to_dict()["db_id"] == "db"
