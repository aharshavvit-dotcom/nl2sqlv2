from __future__ import annotations

import json
from pathlib import Path

from datasets.bird_adapter import BirdAdapter
from datasets.dataset_loader import DatasetLoader


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
