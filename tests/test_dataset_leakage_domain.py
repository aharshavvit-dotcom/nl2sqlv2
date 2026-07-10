from __future__ import annotations

from dataset_training.leakage_checker import DatasetLeakageChecker


def test_query_ir_leakage_blocks_strict_pass() -> None:
    query_ir = {"intent": "show_records", "base_table": "orders", "required_tables": ["orders"]}
    splits = {
        "train": [{"example_id": "t1", "db_id": "db1", "question": "show orders", "source_sql": "select id from orders", "query_ir": query_ir}],
        "validation": [{"example_id": "v1", "db_id": "db2", "question": "display order rows", "source_sql": "select order_id from sales_orders", "query_ir": query_ir}],
    }

    report = DatasetLeakageChecker().run_all_checks(splits)

    assert report["has_query_ir_leakage"] is True
    assert report["strict_passed"] is False


def test_generic_template_overlap_is_reported_but_not_blocking() -> None:
    splits = {
        "train": [{
            "example_id": "t1",
            "db_id": "db1",
            "question": "list customers",
            "source_sql": "select id from customers",
            "schema": {"tables": {"customers": ["id"]}},
            "query_ir": {"intent": "show_records", "base_table": "customers"},
        }],
        "validation": [{
            "example_id": "v1",
            "db_id": "db2",
            "question": "list customers",
            "source_sql": "select account_id from accounts",
            "schema": {"tables": {"accounts": ["account_id"]}},
            "query_ir": {"intent": "show_records", "base_table": "accounts"},
        }],
    }

    report = DatasetLeakageChecker().run_all_checks(splits)

    assert report["generic_template_overlap_count"] == 1
    assert report["has_question_leakage"] is False
    assert report["strict_passed"] is True


def test_parent_child_transitive_leakage_blocks() -> None:
    splits = {
        "train": [
            {"example_id": "root", "db_id": "db1", "question": "root", "source_sql": "select 1"},
            {"example_id": "child", "db_id": "db1", "question": "child", "source_sql": "select 2", "metadata": {"original_example_id": "root"}},
        ],
        "validation": [
            {"example_id": "grandchild", "db_id": "db2", "question": "grandchild", "source_sql": "select 3", "metadata": {"original_example_id": "child"}},
        ],
    }

    report = DatasetLeakageChecker().run_all_checks(splits)

    assert report["has_parent_child_violations"] is True
    assert report["strict_passed"] is False
