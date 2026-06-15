from __future__ import annotations

from pathlib import Path

from datasets.models import DatabaseSchema, Text2SQLExample
from training_ir.build_ir_training_data import build_ir_training_data
from training_ir.validate_ir_corpus import validate_ir_corpus


class OneRowLoader:
    def load(self, dataset_name: str, max_examples: int | None = None):
        schema = DatabaseSchema(
            db_id="shop",
            dataset_name="mock",
            tables={"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}},
            serialized_schema="tables: orders(order_id, amount)",
        )
        return [
            Text2SQLExample(
                example_id="ok",
                dataset_name="mock",
                db_id="shop",
                question="total sales",
                sql="SELECT SUM(orders.amount) AS revenue FROM orders LIMIT 100",
                split="train",
            )
        ], {"shop": schema}


def test_validate_ir_corpus_accepts_generated_training_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    build_ir_training_data(["mock"], output_dir=output_dir, artifact_dir=artifact_dir, loader=OneRowLoader())

    report = validate_ir_corpus(output_dir / "ir_test_examples.jsonl", artifact_dir / "validation.json")

    assert report["invalid_examples"] == 0
    assert (artifact_dir / "validation.json").exists()

