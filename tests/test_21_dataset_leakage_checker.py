from __future__ import annotations

from dataset_training.leakage_checker import DatasetLeakageChecker


def test_database_leakage_detected_for_train_unseen_overlap() -> None:
    splits = {
        "train": [{"db_id": "db1", "question": "q1", "source_sql": "select 1"}],
        "validation": [],
        "test": [],
        "unseen_db_test": [{"db_id": "db1", "question": "q2", "source_sql": "select 2"}],
    }

    report = DatasetLeakageChecker().run_all_checks(splits)

    assert report["has_database_leakage"] is True
    assert report["passed"] is False


def test_question_leakage_detected() -> None:
    splits = {
        "train": [{"db_id": "db1", "question": "List users", "source_sql": "select 1"}],
        "validation": [{"db_id": "db2", "question": "list   users", "source_sql": "select 2"}],
        "test": [],
        "unseen_db_test": [],
    }

    report = DatasetLeakageChecker().run_all_checks(splits)

    assert report["has_question_leakage"] is True
    assert report["question_overlap_count"] == 1


def test_clean_split_passes() -> None:
    splits = {
        "train": [{"db_id": "db1", "question": "List users", "source_sql": "select 1"}],
        "validation": [{"db_id": "db2", "question": "Count berths", "source_sql": "select 2"}],
        "test": [],
        "unseen_db_test": [{"db_id": "db3", "question": "List assignments", "source_sql": "select 3"}],
    }

    assert DatasetLeakageChecker().run_all_checks(splits)["passed"] is True
