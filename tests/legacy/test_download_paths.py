"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

from scripts import dataset_paths


def test_dataset_paths_are_project_relative() -> None:
    assert dataset_paths.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert dataset_paths.get_dataset_dir("wikisql").name == "wikisql"
    assert dataset_paths.get_dataset_dir("bird-mini").name in {"mini_dev", "MINIDEV", "mini_dev_hf"}
    assert "train.jsonl" in dataset_paths.expected_files_for_dataset("wikisql")


def test_ensure_dataset_dirs_creates_expected_folders() -> None:
    dataset_paths.ensure_dataset_dirs()
    assert dataset_paths.RAW_DATA_DIR.exists()
    assert dataset_paths.PROCESSED_DATA_DIR.exists()
    assert dataset_paths.BIRD_MINI_DEV_DIR.exists()
