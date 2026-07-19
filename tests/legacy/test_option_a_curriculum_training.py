"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json

from training_ir.train_option_a_curriculum import train_option_a_curriculum


def test_option_a_curriculum_training_saves_bundle(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    rows = [_row("wikisql:x1", "How many orders?"), _row("spider:x2", "Count orders")]
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    validation_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    report = train_option_a_curriculum(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=tmp_path / "model",
        epochs_per_phase=1,
        batch_size=1,
        max_examples_per_phase=2,
    )

    assert report["phases"]
    assert (tmp_path / "model" / "model.pt").exists()
    assert (tmp_path / "model" / "curriculum_metrics.json").exists()


def _row(example_id: str, question: str) -> dict:
    return {
        "example_id": example_id,
        "question": question,
        "serialized_schema": "tables: orders(order_id, amount)",
        "query_ir": {
            "intent": "count_records",
            "template_id": "count_records",
            "base_table": "orders",
            "required_tables": ["orders"],
            "metrics": [{"aggregation": "COUNT", "table": "orders", "column": "*", "expression": "*"}],
            "dimensions": [],
            "filters": [],
            "date_filters": [],
            "order_by": [],
            "limit": 100,
        },
    }

