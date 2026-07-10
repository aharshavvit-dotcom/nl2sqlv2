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


def test_generic_count_overlap_is_reported_but_not_blocking() -> None:
    splits = {
        "train": [
            {
                "example_id": "t1",
                "db_id": "db1",
                "question": "Count the number of customers.",
                "source_sql": "SELECT count(*) FROM Customers",
                "schema": {"tables": {"Customers": {"columns": {"id": {}, "name": {}}}}},
                "query_ir": {"query_ir_id": "t1", "intent": "count_records", "base_table": "Customers"},
            }
        ],
        "validation": [
            {
                "example_id": "v1",
                "db_id": "db2",
                "question": "count the number of customers.",
                "source_sql": "SELECT count(*) FROM Customers",
                "schema": {"tables": {"Customers": {"columns": {"customer_id": {}, "region": {}}}}},
                "query_ir": {"query_ir_id": "v1", "intent": "count_records", "base_table": "Customers"},
            }
        ],
        "test": [],
        "unseen_db_test": [],
    }

    report = DatasetLeakageChecker().run_all_checks(splits)

    assert report["generic_template_overlap_count"] == 1
    assert report["generic_sql_overlap_count"] == 1
    assert report["generic_sql_ast_overlap_count"] == 1
    assert report["has_question_leakage"] is False
    assert report["has_sql_leakage"] is False
    assert report["has_sql_ast_leakage"] is False
    assert report["has_near_duplicate_leakage"] is False
    assert report["strict_passed"] is True


def test_clean_split_passes() -> None:
    splits = {
        "train": [{"db_id": "db1", "question": "List users", "source_sql": "select 1"}],
        "validation": [{"db_id": "db2", "question": "Count berths", "source_sql": "select 2"}],
        "test": [],
        "unseen_db_test": [{"db_id": "db3", "question": "List assignments", "source_sql": "select 3"}],
    }

    assert DatasetLeakageChecker().run_all_checks(splits)["passed"] is True
