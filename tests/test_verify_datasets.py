from __future__ import annotations

from pathlib import Path

from scripts.verify_datasets import verify_bird_full, verify_spider, verify_wikisql


def test_verify_datasets_handles_missing_paths(tmp_path: Path) -> None:
    wiki = verify_wikisql(tmp_path / "wikisql")
    spider = verify_spider(tmp_path / "spider")

    assert wiki.status == "missing"
    assert wiki.example_count == 0
    assert spider.status == "missing"


def test_verify_bird_full_reports_partial_zip(tmp_path: Path) -> None:
    raw = tmp_path / "bird-full"
    raw.mkdir()
    (raw / "train.zip").write_bytes(b"")

    result = verify_bird_full(raw)

    assert result.status == "incomplete"
    assert "train.zip" in result.notes
