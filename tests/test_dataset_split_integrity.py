from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_training.split_manager import DatasetSplitManager


def _row(db_id: str, *, source_split: str | None = None) -> dict:
    row = {
        "example_id": f"{db_id}_1",
        "dataset_name": "mock",
        "db_id": db_id,
        "database_id": db_id,
        "question": f"show records for {db_id}",
        "source_sql": f"SELECT id FROM {db_id}",
        "query_ir": {"intent": "show_records", "base_table": db_id, "required_tables": [db_id]},
    }
    if source_split:
        row["source_split"] = source_split
        row["split"] = source_split
        row["eligible_for_training"] = source_split == "train"
    return row


def _required_rows() -> list[dict]:
    return [
        _row("train_db"),
        _row("dev_db"),
        _row("model_select_db"),
        _row("frozen_db"),
        _row("unseen_db"),
        _row("controlled_db"),
    ]


def _manifest_for(rows: list[dict], manager: DatasetSplitManager) -> dict:
    return {
        "split_schema_version": "1.0",
        "split_version": "unit",
        "created_at": "2026-07-09T00:00:00+00:00",
        "random_seed": 42,
        "algorithm": "group_multilabel_stratification",
        "algorithm_version": "1.0",
        "group_key": "database_id",
        "dataset_hashes": manager._dataset_hashes({"all": rows}),
        "train_db_ids": ["train_db"],
        "development_validation_db_ids": ["dev_db"],
        "model_selection_validation_db_ids": ["model_select_db"],
        "frozen_semantic_test_db_ids": ["frozen_db"],
        "unseen_database_test_db_ids": ["unseen_db"],
        "controlled_execution_test_db_ids": ["controlled_db"],
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_manifest_applies_without_defaulting_unknowns_to_train(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    path = _write_manifest(tmp_path, _manifest_for(rows, manager))

    splits = manager.apply_manifest_split(rows, path)

    assert [row["db_id"] for row in splits["train"]] == ["train_db"]
    assert [row["db_id"] for row in splits["validation"]] == ["dev_db"]
    assert [row["db_id"] for row in splits["controlled_execution_test"]] == ["controlled_db"]
    assert all(row["internal_split"] for rows_for_split in splits.values() for row in rows_for_split)


def test_placeholder_manifest_rejected() -> None:
    rows = [_row(db_id) for db_id in ["db_a", "db_b", "db_c", "db_d", "db_e", "db_f"]]
    fixture = Path(__file__).parent / "fixtures" / "splits" / "placeholder_split_manifest.json"

    with pytest.raises(ValueError, match="placeholder database IDs"):
        DatasetSplitManager().apply_manifest_split(rows, fixture)


def test_unknown_manifest_database_rejected(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    manifest = _manifest_for(rows, manager)
    manifest["controlled_execution_test_db_ids"] = ["missing_db"]
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unknown databases"):
        manager.apply_manifest_split(rows, path)


def test_database_absent_from_manifest_rejected(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    manifest = _manifest_for(rows, manager)
    manifest["controlled_execution_test_db_ids"] = []
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="absent from split manifest|empty required splits"):
        manager.apply_manifest_split(rows, path)


def test_database_in_multiple_splits_rejected(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    manifest = _manifest_for(rows, manager)
    manifest["development_validation_db_ids"] = ["train_db"]
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="multiple splits"):
        manager.apply_manifest_split(rows, path)


def test_dataset_hash_mismatch_rejected(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    manifest = _manifest_for(rows, manager)
    manifest["dataset_hashes"] = {"mock": "bad"}
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="dataset_hashes mismatch"):
        manager.apply_manifest_split(rows, path)


def test_official_source_test_record_cannot_enter_training(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    rows[0] = _row("train_db", source_split="test")
    manifest = _manifest_for(rows, manager)
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="Source lineage forbids training assignment"):
        manager.apply_manifest_split(rows, path)


def test_force_create_new_version_rebuilds_stale_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DatasetSplitManager(split_version="unit", force_create_new_version=True, split_dir=tmp_path)
    legacy_rows = _required_rows()
    _write_manifest(tmp_path / "unit", _manifest_for(legacy_rows, manager))

    def _fail_manifest(*args, **kwargs):
        raise AssertionError("stale manifest should not be applied")

    monkeypatch.setattr(manager, "apply_manifest_split", _fail_manifest)

    fresh_rows = [_row("fresh_db"), _row("train_db"), _row("dev_db")]
    splits = manager.split_by_database(fresh_rows)

    assert any(row.get("db_id") == "fresh_db" for split_name in ["train", "validation", "test", "unseen_db_test"] for row in splits[split_name])


def test_validation_source_rows_stay_out_of_training_split(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    row = _row("db_one", source_split="validation")
    extra_rows = [_row(db_id) for db_id in ["train_db", "frozen_db", "unseen_db", "controlled_db"]]
    manifest = {
        "split_schema_version": "1.0",
        "split_version": "unit",
        "created_at": "2026-07-09T00:00:00+00:00",
        "random_seed": 42,
        "algorithm": "group_multilabel_stratification",
        "algorithm_version": "1.0",
        "group_key": "database_id",
        "dataset_hashes": manager._dataset_hashes({"all": [row, *extra_rows]}),
        "train_db_ids": ["train_db"],
        "development_validation_db_ids": ["db_one"],
        "model_selection_validation_db_ids": [],
        "frozen_semantic_test_db_ids": ["frozen_db"],
        "unseen_database_test_db_ids": ["unseen_db"],
        "controlled_execution_test_db_ids": ["controlled_db"],
    }
    path = _write_manifest(tmp_path, manifest)

    splits = manager.apply_manifest_split([row, *extra_rows], path)

    assert all(item.get("db_id") != "db_one" for item in splits["train"])
    assert any(item.get("db_id") == "db_one" for item in splits["validation"])


def test_manifest_immutability_requires_explicit_force(tmp_path: Path) -> None:
    manager = DatasetSplitManager(split_version="unit")
    rows = _required_rows()
    splits = {
        "train": [rows[0]],
        "validation": [rows[1]],
        "model_selection_validation": [rows[2]],
        "test": [rows[3]],
        "unseen_db_test": [rows[4]],
        "controlled_execution_test": [rows[5]],
        "unsupported": [],
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        manager.save_manifest_split(splits, manifest_path)
