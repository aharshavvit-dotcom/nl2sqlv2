"""
Purpose: Verifies data unit behaviour consolidated from fragmented test files.
Required because: Dataset split, leakage, corpus, scale and verification tests protect the dataset pipeline.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_20_dataset_split_manager.py
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


# Source: tests/test_21_dataset_leakage_checker.py
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


# Source: tests/test_22_generic_ir_corpus_builder.py
from pathlib import Path

from dataset_training import DatasetRegistry, DatasetSplitManager, GenericIRCorpusBuilder
from datasets.models import DatabaseSchema, Text2SQLExample


class FakeRegistry(DatasetRegistry):
    def __init__(self):
        super().__init__(root_dir="unused")

    def validate_dataset_presence(self, dataset_names: list[str]) -> dict:
        return {name: {"available": True, "paths": {}, "missing_files": []} for name in dataset_names}

    def load_examples(self, dataset_names: list[str], max_examples: int | None = None):
        schema = DatabaseSchema(
            db_id="db_users",
            dataset_name="mock",
            tables={"users": {"columns": {"id": {}, "name": {}, "role": {}}}},
            serialized_schema="tables: users(id, name, role)",
        )
        examples = [
            Text2SQLExample(
                example_id="ex_supported",
                dataset_name="mock",
                db_id="db_users",
                question="list users",
                sql="SELECT users.id, users.name FROM users LIMIT 100",
                split="train",
            ),
            Text2SQLExample(
                example_id="ex_unsupported",
                dataset_name="mock",
                db_id="db_users",
                question="nested users",
                sql="SELECT id FROM users WHERE id IN (SELECT id FROM users) LIMIT 100",
                split="train",
            ),
        ]
        return examples[:max_examples], {"db_users": schema}


class MultiDatasetRegistry(DatasetRegistry):
    def __init__(self):
        super().__init__(root_dir="unused")

    def validate_dataset_presence(self, dataset_names: list[str]) -> dict:
        return {name: {"available": True, "paths": {}, "missing_files": []} for name in dataset_names}

    def load_examples(self, dataset_names: list[str], max_examples: int | None = None):
        dataset_name = dataset_names[0]
        schema = DatabaseSchema(
            db_id=f"{dataset_name}_db",
            dataset_name=dataset_name,
            tables={"users": {"columns": {"id": {}, "name": {}}}},
            serialized_schema="tables: users(id, name)",
        )
        examples = [
            Text2SQLExample(
                example_id=f"{dataset_name}_{idx}",
                dataset_name=dataset_name,
                db_id=f"{dataset_name}_db",
                question="list users",
                sql="SELECT users.id, users.name FROM users LIMIT 100",
                split="train",
            )
            for idx in range(3)
        ]
        return examples[:max_examples], {f"{dataset_name}_db": schema}


class RenameRegistry(DatasetRegistry):
    def __init__(self):
        super().__init__(root_dir="unused")

    def validate_dataset_presence(self, dataset_names: list[str]) -> dict:
        return {name: {"available": True, "paths": {}, "missing_files": []} for name in dataset_names}

    def load_examples(self, dataset_names: list[str], max_examples: int | None = None):
        schema = DatabaseSchema(
            db_id="orders_db",
            dataset_name="mock",
            tables={"orders": {"columns": {"id": {}, "amount": {}}}},
            serialized_schema="tables: orders(id, amount)",
        )
        examples = [
            Text2SQLExample(
                example_id="orders_supported",
                dataset_name="mock",
                db_id="orders_db",
                question="list orders",
                sql="SELECT orders.id, orders.amount FROM orders LIMIT 100",
                split="train",
            ),
        ]
        return examples[:max_examples], {"orders_db": schema}


def test_generic_ir_corpus_builder_writes_splits_and_reports(tmp_path: Path) -> None:
    output = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    report = GenericIRCorpusBuilder(
        dataset_registry=FakeRegistry(),
        split_manager=DatasetSplitManager(seed=1, unseen_db_test_ratio=0.0),
        sql_to_ir_converter=None,
        quality_filter=None,
    ).build(["mock"], max_examples=None, output_dir=str(output), artifact_dir=str(artifacts))

    assert (output / "generic_ir_train.jsonl").exists()
    assert (output / "generic_ir_validation.jsonl").exists()
    assert (output / "generic_ir_test.jsonl").exists()
    assert (output / "generic_ir_unseen_db_test.jsonl").exists()
    assert (output / "generic_ir_unsupported.jsonl").exists()
    assert (artifacts / "corpus_quality_report.json").exists()
    assert report["corpus_quality_report"]["supported_examples"] == 1
    assert report["corpus_quality_report"]["unsupported_examples"] == 1


def test_generic_ir_corpus_builder_renders_schema_renaming_augmentations(tmp_path: Path) -> None:
    report = GenericIRCorpusBuilder(
        dataset_registry=RenameRegistry(),
        split_manager=DatasetSplitManager(seed=1, unseen_db_test_ratio=0.0),
        sql_to_ir_converter=None,
        quality_filter=None,
    ).build(
        ["mock"],
        max_examples=None,
        output_dir=str(tmp_path / "processed"),
        artifact_dir=str(tmp_path / "artifacts"),
        schema_renaming={
            "enabled": True,
            "multiplier": 1,
            "modes": ["neutral_names"],
        },
    )

    assert report["augmentation_report"]["augmented_examples_count"] == 1


def test_generic_ir_corpus_builder_applies_cap_per_dataset(tmp_path: Path) -> None:
    report = GenericIRCorpusBuilder(
        dataset_registry=MultiDatasetRegistry(),
        split_manager=DatasetSplitManager(seed=1, unseen_db_test_ratio=0.0),
    ).build(
        ["wikisql", "spider"],
        max_examples=1,
        output_dir=str(tmp_path / "processed"),
        artifact_dir=str(tmp_path / "artifacts"),
        min_converted_examples_required={"wikisql": 1, "spider": 1},
    )

    by_dataset = report["dataset_contribution_report"]["by_dataset"]
    assert by_dataset["wikisql"]["loaded_examples"] == 1
    assert by_dataset["spider"]["loaded_examples"] == 1
    assert report["dataset_contribution_report"]["full_training_dataset_minimums_passed"] is True


# Source: tests/test_24_dataset_scale_evaluator.py
from copy import deepcopy

import pytest

from dataset_training.dataset_evaluator import DatasetScaleEvaluator


def test_dataset_scale_evaluator_reports_core_metrics() -> None:
    gold = {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": [], "dimensions": [], "filters": [], "date_filters": []}
    bad = deepcopy(gold)
    bad["joins"] = [{"condition": "assignments.user_id = users.id"}]
    row = {
        "example_id": "ex1",
        "dataset_name": "mock",
        "db_id": "db1",
        "complexity": "simple",
        "question": "list users",
        "query_ir": gold,
        "predicted_query_ir": bad,
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.95,
        "prediction_latency_ms": 12.0,
        "retrieval_scores": [0.9, 0.7],
    }

    report = DatasetScaleEvaluator().evaluate_model("mock_model", [row])

    assert report["summary"]["intent_accuracy_rate"] == 1.0
    assert report["summary"]["join_accuracy_rate"] == 0.0
    assert report["summary"]["unnecessary_join_rate"] == 1.0
    assert report["summary"]["sql_validation_rate"] == 1.0
    assert report["by_intent"]["show_records"]["total_examples"] == 1
    assert report["classification_metrics"]["intent"]["macro_f1"] == 1.0
    assert report["classification_metrics"]["join_decision"]["macro_f1"] == 0.0
    assert report["confusion_matrices"]["join_decision"]["no_join_required"]["unnecessary_join"] == 1
    assert report["percentiles"]["prediction_latency_ms_p95"] == 12.0
    assert report["percentiles"]["retrieval_margin_p50"] == pytest.approx(0.2)
    assert report["calibration"]["sample_count"] == 1
    assert report["evaluation_mode"] == "real_model_predictions"
    assert report["is_valid_for_quality_gate"] is True


def test_dataset_scale_evaluator_refuses_implicit_gold_replay() -> None:
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
    }

    with pytest.raises(ValueError, match="requires real predicted_query_ir"):
        DatasetScaleEvaluator().evaluate_model("mock_model", [row])


def test_dataset_scale_evaluator_labels_explicit_gold_replay() -> None:
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
    }

    report = DatasetScaleEvaluator().evaluate_model(
        "debug_gold",
        [row],
        evaluation_mode="explicit_gold_replay_baseline",
    )

    assert report["gold_replay_used"] is True
    assert report["is_valid_for_quality_gate"] is False


def test_evaluator_invalid_when_zero_real_predictions() -> None:
    """Even with predicted_query_ir present, if real_predictions_generated sums to 0 the report is invalid."""
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "predicted_query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.9,
        "prediction_latency_ms": 10.0,
    }

    report = DatasetScaleEvaluator().evaluate_model(
        "mock_model",
        [row],
        evaluation_mode="real_model_predictions",
        predictor_used=False,
    )

    assert report["is_valid_for_quality_gate"] is False


def test_evaluator_invalid_when_predictor_not_used() -> None:
    """When predictor_used=False is explicitly passed, the report must be invalid."""
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "predicted_query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.95,
        "prediction_latency_ms": 12.0,
    }

    report = DatasetScaleEvaluator().evaluate_model(
        "mock_model",
        [row],
        evaluation_mode="real_model_predictions",
        predictor_used=False,
    )

    assert report["predictor_used"] is False
    assert report["is_valid_for_quality_gate"] is False


def test_evaluator_invalid_when_rows_evaluated_zero() -> None:
    """Empty rows list → is_valid_for_quality_gate must be False."""
    report = DatasetScaleEvaluator().evaluate_model(
        "mock_model",
        [],
        evaluation_mode="real_model_predictions",
        predictor_used=True,
    )

    assert report["rows_evaluated"] == 0
    assert report["is_valid_for_quality_gate"] is False


def test_per_example_contains_bootstrap_promotion_fields() -> None:
    """per_example must contain simple_query_pass, gold_comparison_score, unseen_db_sql_valid
    for bootstrap promotion to work (see promotion_policy.py)."""
    gold = {"intent": "show_records", "base_table": "users", "joins": []}
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": gold,
        "predicted_query_ir": gold,
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.95,
        "prediction_latency_ms": 10.0,
    }

    # Gold schema mode → unseen_db_sql_valid should be None
    report = DatasetScaleEvaluator().evaluate_model("mock_model", [row], schema_mode="gold")
    pe = report["per_example"][0]
    assert "simple_query_pass" in pe
    assert "gold_comparison_score" in pe
    assert "unseen_db_sql_valid" in pe
    assert pe["unseen_db_sql_valid"] is None  # Not unseen_db mode
    # simple_query_pass should be True for a correct simple query
    assert pe["simple_query_pass"] is True

    # Unseen-DB schema mode → unseen_db_sql_valid should be bool
    report_unseen = DatasetScaleEvaluator().evaluate_model("mock_model", [row], schema_mode="unseen_db")
    pe_unseen = report_unseen["per_example"][0]
    assert isinstance(pe_unseen["unseen_db_sql_valid"], bool)
    assert pe_unseen["gold_comparison_score"] >= 0.0


def test_simple_query_pass_show_records_correct() -> None:
    """show_records with correct table and no joins → simple_query_pass = True."""
    gold = {"intent": "show_records", "base_table": "users", "joins": []}
    row = {
        "example_id": "sq1", "question": "list users",
        "query_ir": gold, "predicted_query_ir": gold,
        "ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True},
        "confidence": 0.9, "prediction_latency_ms": 5.0,
    }
    report = DatasetScaleEvaluator().evaluate_model("m", [row])
    assert report["per_example"][0]["simple_query_pass"] is True


def test_simple_query_pass_count_records_correct() -> None:
    """count_records with correct table and no joins → simple_query_pass = True."""
    gold = {"intent": "count_records", "base_table": "orders", "joins": []}
    row = {
        "example_id": "sq2", "question": "count orders",
        "query_ir": gold, "predicted_query_ir": gold,
        "ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True},
        "confidence": 0.9, "prediction_latency_ms": 5.0,
    }
    report = DatasetScaleEvaluator().evaluate_model("m", [row])
    assert report["per_example"][0]["simple_query_pass"] is True


def test_simple_query_pass_false_with_unnecessary_join() -> None:
    """Simple gold query but prediction adds an unnecessary join → simple_query_pass = False."""
    gold = {"intent": "show_records", "base_table": "users", "joins": []}
    pred = {"intent": "show_records", "base_table": "users", "joins": [{"condition": "a.id=b.id"}]}
    row = {
        "example_id": "sq3", "question": "list users",
        "query_ir": gold, "predicted_query_ir": pred,
        "ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True},
        "confidence": 0.9, "prediction_latency_ms": 5.0,
    }
    report = DatasetScaleEvaluator().evaluate_model("m", [row])
    assert report["per_example"][0]["simple_query_pass"] is False


def test_simple_query_pass_false_with_wrong_table() -> None:
    """Simple gold query but prediction has wrong base_table → simple_query_pass = False."""
    gold = {"intent": "show_records", "base_table": "users", "joins": []}
    pred = {"intent": "show_records", "base_table": "orders", "joins": []}
    row = {
        "example_id": "sq4", "question": "list users",
        "query_ir": gold, "predicted_query_ir": pred,
        "ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True},
        "confidence": 0.9, "prediction_latency_ms": 5.0,
    }
    report = DatasetScaleEvaluator().evaluate_model("m", [row])
    assert report["per_example"][0]["simple_query_pass"] is False


def test_simple_query_pass_none_for_non_simple_query() -> None:
    """Non-simple gold query (has joins) → simple_query_pass = None (excluded from rate)."""
    gold = {"intent": "joined_records", "base_table": "users", "joins": [{"condition": "a.id=b.uid"}]}
    row = {
        "example_id": "sq5", "question": "users with orders",
        "query_ir": gold, "predicted_query_ir": gold,
        "ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True},
        "confidence": 0.9, "prediction_latency_ms": 5.0,
    }
    report = DatasetScaleEvaluator().evaluate_model("m", [row])
    assert report["per_example"][0]["simple_query_pass"] is None


def test_simple_query_pass_none_for_aggregation_intent() -> None:
    """Aggregation intent (not in simple set) → simple_query_pass = None."""
    gold = {"intent": "metric_summary", "base_table": "sales", "joins": []}
    row = {
        "example_id": "sq6", "question": "total sales",
        "query_ir": gold, "predicted_query_ir": gold,
        "ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True},
        "confidence": 0.9, "prediction_latency_ms": 5.0,
    }
    report = DatasetScaleEvaluator().evaluate_model("m", [row])
    assert report["per_example"][0]["simple_query_pass"] is None


# Source: tests/test_dataset_leakage_domain.py
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


# Source: tests/test_dataset_split_integrity.py
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
    fixture = Path(__file__).parents[2] / "fixtures" / "splits" / "placeholder_split_manifest.json"

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


def test_manifest_records_parent_split_and_regeneration_reason(tmp_path: Path) -> None:
    manager = DatasetSplitManager(
        split_version="unit",
        split_dir=tmp_path,
        parent_split_version="semantic_v1",
        regeneration_reason="membership changed",
    )
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

    manager.save_manifest_split(splits, manager.get_manifest_path())
    manifest = json.loads(manager.get_manifest_path().read_text(encoding="utf-8"))

    assert manifest["split_version"] == "unit"
    assert manifest["parent_split_version"] == "semantic_v1"
    assert manifest["regeneration_reason"] == "membership changed"
    assert manifest["force_create_new_version"] is False


# Source: tests/test_verify_datasets.py
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


def test_verify_bird_full_ready_when_prepared(tmp_path: Path) -> None:
    raw = tmp_path / "bird-full"
    raw.mkdir()
    for name in ["train.json", "validation.json", "test.json"]:
        (raw / name).write_text("[]", encoding="utf-8")
    (raw / "train_tables.json").write_text("[]", encoding="utf-8")
    (raw / "dev_tables.json").write_text("[]", encoding="utf-8")

    result = verify_bird_full(raw)

    assert result.status == "ready"
    assert result.example_count == 0


# Source: tests/test_sql_partial_supervision.py
from capabilities import SQLCapabilityExtractor
from ir.sql_to_ir_converter import SQLToIRConverter


def test_partial_supervision_retains_labels_for_unsupported_window_query() -> None:
    sql = (
        "SELECT customer_id, ROW_NUMBER() OVER "
        "(PARTITION BY region ORDER BY amount DESC) AS rn FROM orders"
    )
    result = SQLToIRConverter().convert("rank orders", sql, schema=None)
    annotation = SQLCapabilityExtractor().with_conversion_result(
        SQLCapabilityExtractor().extract(sql),
        result,
    )
    partial = annotation.partial_supervision

    assert result["success"] is False
    assert "WINDOW_ROW_NUMBER" in annotation.required_capabilities
    assert partial.full_query_ir_supported is False
    assert partial.unsupported_reason == "window_function"
    assert partial.referenced_tables == ["orders"]
    assert "customer_id" in partial.selected_columns
    assert partial.window_functions[0].function == "ROW_NUMBER"
    assert annotation.task_masks.window == 1
    assert annotation.task_masks.full_query_ir == 0


def test_partial_supervision_captures_correlated_subquery_details() -> None:
    sql = (
        "SELECT c.id FROM customers c "
        "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)"
    )
    partial = SQLCapabilityExtractor().extract(sql).partial_supervision

    assert "CORRELATED_SUBQUERY" in partial.subquery_types
    assert partial.subquery_depth >= 1
    assert partial.correlated_subqueries
    assert "c.id" in partial.correlated_subqueries[0].correlated_columns
    assert "EQ" in partial.correlated_subqueries[0].correlation_operators


def test_partial_supervision_captures_set_operation_branches() -> None:
    partial = SQLCapabilityExtractor().extract("SELECT id FROM a UNION ALL SELECT id FROM b").partial_supervision

    assert partial.set_operation == "UNION_ALL"
    assert len(partial.set_operation_branches) == 2
    assert partial.set_operation_branches[0].required_capabilities
