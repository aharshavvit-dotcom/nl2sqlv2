from __future__ import annotations

from dataset_training.split_manager import DatasetSplitManager


def _rows() -> list[dict]:
    rows = []
    for db_id in ["db_a", "db_b", "db_c", "db_d", "db_e", "db_f"]:
        for index in range(3):
            rows.append(
                {
                    "example_id": f"{db_id}_{index}",
                    "dataset_name": "mock",
                    "db_id": db_id,
                    "question": f"question {db_id} {index}",
                    "source_sql": f"SELECT id FROM {db_id} LIMIT 100",
                    "query_ir": {"intent": "show_records", "base_table": db_id, "required_tables": [db_id]},
                }
            )
    return rows


def test_database_level_splits_are_created_without_unseen_leakage() -> None:
    splits = DatasetSplitManager(seed=7, unseen_db_test_ratio=0.2).split_by_database(_rows())

    assert set(splits) == {"train", "validation", "test", "unseen_db_test", "unsupported"}
    assert splits["train"]
    assert splits["test"]
    assert splits["unseen_db_test"]
    train_dbs = {row["db_id"] for row in splits["train"]}
    validation_dbs = {row["db_id"] for row in splits["validation"]}
    unseen_dbs = {row["db_id"] for row in splits["unseen_db_test"]}
    test_dbs = {row["db_id"] for row in splits["test"]}
    assert unseen_dbs.isdisjoint(train_dbs | validation_dbs)
    assert train_dbs.isdisjoint(validation_dbs | test_dbs)
    assert validation_dbs.isdisjoint(test_dbs)
    assert sum(len(rows) for rows in splits.values()) == len(_rows())


def test_split_by_dataset_and_database_preserves_counts() -> None:
    rows = _rows()
    rows.extend([{**row, "dataset_name": "mock2", "example_id": "m2_" + row["example_id"]} for row in _rows()])

    splits = DatasetSplitManager(seed=3).split_by_dataset_and_database(rows)

    assert sum(len(values) for values in splits.values()) == len(rows)
    assert {row["dataset_name"] for rows_for_split in splits.values() for row in rows_for_split} == {"mock", "mock2"}
