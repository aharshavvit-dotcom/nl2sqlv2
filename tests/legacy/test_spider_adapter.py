"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets.spider_adapter import SpiderAdapter


def test_spider_adapter_loads_mock_files(tmp_path: Path) -> None:
    raw = tmp_path / "spider"
    (raw / "database" / "shop").mkdir(parents=True)
    (raw / "database" / "shop" / "shop.sqlite").write_bytes(b"")
    (raw / "tables.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "shop",
                    "table_names_original": ["orders"],
                    "column_names_original": [[-1, "*"], [0, "order_id"], [0, "amount"]],
                    "column_types": ["*", "number", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    (raw / "train_spider.json").write_text(
        json.dumps([{"db_id": "shop", "question": "total sales", "query": "SELECT SUM(amount) FROM orders"}]),
        encoding="utf-8",
    )
    (raw / "dev.json").write_text("[]", encoding="utf-8")

    adapter = SpiderAdapter()
    schemas = adapter.load_schemas(raw)
    examples = adapter.load_examples(raw)

    assert "shop" in schemas
    assert schemas["shop"].db_path is not None
    assert examples[0].question == "total sales"
    assert examples[0].split == "train"
    assert examples[0].tables == ["orders"]
