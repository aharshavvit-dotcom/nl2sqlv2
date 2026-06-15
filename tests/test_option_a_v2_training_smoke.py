from __future__ import annotations

import json

from training_ir.train_option_a_v2_model import train_option_a_v2_model


def test_option_a_v2_training_smoke_saves_bundle(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    rows = [_row("x1", "How many orders?"), _row("x2", "Count orders")]
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    validation_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    report = train_option_a_v2_model(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=tmp_path / "model",
        epochs=1,
        batch_size=1,
        max_examples=2,
        model_overrides={"embedding_dim": 8, "hidden_dim": 8, "candidate_hidden_dim": 6, "max_tables": 4, "max_columns": 8, "max_schema_len": 32},
    )

    assert report["model_version"] == "option_a_v2"
    assert (tmp_path / "model" / "model.pt").exists()
    assert (tmp_path / "model" / "option_a_calibration.json").exists()


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
            "metrics": [{"name": "record_count", "aggregation": "COUNT", "table": "orders", "column": "*", "expression": "*", "alias": "record_count"}],
            "dimensions": [],
            "filters": [],
            "date_filters": [],
            "order_by": [],
            "limit": 100,
            "metadata": {"validation_context": {"schema_context": {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}}},
        },
    }
