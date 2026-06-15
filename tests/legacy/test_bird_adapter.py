from __future__ import annotations

import json
from pathlib import Path

from datasets.bird_adapter import BirdAdapter
from datasets.dataset_loader import DatasetLoader
from scripts.prepare_bird_full import prepare_bird_full


def test_bird_adapter_loads_local_json_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "bird"
    raw.mkdir()
    (raw / "mini_dev_sqlite.json").write_text(
        json.dumps(
            [
                {
                    "question": "How many singers?",
                    "SQL": "SELECT COUNT(*) FROM singers",
                    "db_id": "music",
                    "difficulty": "simple",
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = BirdAdapter().load_examples(raw, max_examples=1)

    assert len(examples) == 1
    assert examples[0].dataset_name == "bird-mini"
    assert examples[0].split == "validation"
    assert examples[0].sql.startswith("SELECT")


def test_dataset_loader_load_datasets_alias(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    bird = raw_root / "bird" / "mini_dev"
    bird.mkdir(parents=True)
    (bird / "mini_dev_sqlite.json").write_text(
        json.dumps([{"question": "How many singers?", "SQL": "SELECT COUNT(*) FROM singers", "db_id": "music"}]),
        encoding="utf-8",
    )

    examples, schemas = DatasetLoader(raw_root=raw_root).load_datasets(["bird-mini"], max_examples=1)

    assert len(examples) == 1
    assert schemas == {}


def test_bird_full_prepare_creates_train_validation_test_layout(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "bird" / "full"
    train_dir = raw / "train" / "train"
    dev_dir = raw / "dev_20240627"
    train_dir.mkdir(parents=True)
    dev_dir.mkdir(parents=True)
    (train_dir / "train.json").write_text(
        json.dumps(
            [
                {"question": "q1", "SQL": "SELECT COUNT(*) FROM orders", "db_id": "shop"},
                {"question": "q2", "SQL": "SELECT order_id FROM orders", "db_id": "shop"},
            ]
        ),
        encoding="utf-8",
    )
    (train_dir / "train_tables.json").write_text(
        json.dumps([{"db_id": "shop", "table_names_original": ["orders"], "column_names_original": [[-1, "*"], [0, "order_id"]]}]),
        encoding="utf-8",
    )
    (dev_dir / "dev.json").write_text(
        json.dumps(
            [
                {"question_id": 1, "question": "v1", "SQL": "SELECT order_id FROM orders", "db_id": "db_validation"},
                {"question_id": 2, "question": "v2", "SQL": "SELECT order_id FROM orders", "db_id": "db_validation"},
                {"question_id": 3, "question": "t1", "SQL": "SELECT COUNT(*) FROM orders", "db_id": "db_test"},
                {"question_id": 4, "question": "t2", "SQL": "SELECT COUNT(*) FROM orders", "db_id": "db_test"},
            ]
        ),
        encoding="utf-8",
    )
    (dev_dir / "dev_tables.json").write_text(
        json.dumps([{"db_id": "shop", "table_names_original": ["orders"], "column_names_original": [[-1, "*"], [0, "order_id"]]}]),
        encoding="utf-8",
    )
    mac = raw / "train" / "__MACOSX"
    mac.mkdir(parents=True)
    (mac / "._train.json").write_bytes(b"\xb0broken")

    manifest = prepare_bird_full(raw, test_ratio=0.5)
    examples = BirdAdapter().load_examples(raw, split_type="full")

    assert manifest["counts"] == {"train": 2, "validation": 2, "test": 2, "dev_source": 4}
    assert manifest["split_policy"]["test_db_ids"] == ["db_test"]
    assert [example.split for example in examples] == ["train", "train", "validation", "validation", "test", "test"]
    assert {example.dataset_name for example in examples} == {"bird-full"}
    validation_db_ids = {example.db_id for example in examples if example.split == "validation"}
    test_db_ids = {example.db_id for example in examples if example.split == "test"}
    assert validation_db_ids.isdisjoint(test_db_ids)
