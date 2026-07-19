"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets.models import DatabaseSchema, Text2SQLExample
from training_ir.build_ir_training_data import build_ir_training_data


class FakeLoader:
    def load(self, dataset_name: str, max_examples: int | None = None):
        schema = DatabaseSchema(
            db_id="shop",
            dataset_name="mock",
            tables={
                "orders": {
                    "columns": [
                        {"name": "order_id"},
                        {"name": "amount"},
                        {"name": "status"},
                    ]
                }
            },
            serialized_schema="tables: orders(order_id, amount, status)",
        )
        examples = [
            Text2SQLExample(
                example_id="ok",
                dataset_name="mock",
                db_id="shop",
                question="total sales",
                sql="SELECT SUM(orders.amount) AS revenue FROM orders LIMIT 100",
                split="train",
            ),
            Text2SQLExample(
                example_id="bad",
                dataset_name="mock",
                db_id="shop",
                question="nested",
                sql="SELECT orders.order_id FROM orders WHERE orders.amount > (SELECT AVG(orders.amount) FROM orders)",
                split="train",
            ),
        ]
        return examples[:max_examples], {"shop": schema}


def test_build_ir_training_data_writes_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"

    report = build_ir_training_data(
        datasets=["mock"],
        max_examples=10,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        loader=FakeLoader(),
    )

    assert report["successful_examples"] == 1
    assert (output_dir / "ir_training_examples.jsonl").exists()
    assert (output_dir / "ir_validation_examples.jsonl").exists()
    assert (output_dir / "ir_test_examples.jsonl").exists()
    assert (output_dir / "ir_unsupported_examples.jsonl").exists()
    assert (output_dir / "ir_dataset_stats.json").exists()
    assert (artifact_dir / "ir_corpus_report.json").exists()
    unsupported = [json.loads(line) for line in (output_dir / "ir_unsupported_examples.jsonl").read_text().splitlines()]
    assert unsupported[0]["unsupported_reason"] == "nested_query"

