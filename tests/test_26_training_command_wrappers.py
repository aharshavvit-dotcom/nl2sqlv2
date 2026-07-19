"""
Purpose: Protects ir unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _row() -> dict:
    return {
        "example_id": "ex1",
        "dataset_name": "mock",
        "db_id": "db1",
        "question": "list users",
        "source_sql": "SELECT users.id FROM users LIMIT 100",
        "rendered_sql": "SELECT users.id FROM users LIMIT 100",
        "intent": "show_records",
        "template_id": "show_records",
        "schema": {"tables": {"users": {"columns": {"id": {}}}}},
        "query_ir": {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": [], "dimensions": [], "filters": [], "date_filters": []},
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "roundtrip_validation": {"is_valid": True},
    }


def test_training_command_wrappers_with_tiny_files(tmp_path: Path) -> None:
    train = tmp_path / "generic_ir_train.jsonl"
    test = tmp_path / "generic_ir_test.jsonl"
    unseen = tmp_path / "generic_ir_unseen_db_test.jsonl"
    _write_jsonl(train, [_row()])
    _write_jsonl(test, [_row()])
    _write_jsonl(unseen, [_row()])

    commands = [
        [sys.executable, "training/build_retrieval_rag_index.py", "--input", str(train), "--output-dir", str(tmp_path / "rag")],
        [sys.executable, "training/build_hard_negative_corpus.py", "--input", str(train), "--output", str(tmp_path / "hard.jsonl"), "--max-negatives-per-example", "2"],
        [sys.executable, "training/evaluate_generic_models.py", "--test", str(test), "--unseen-db-test", str(unseen), "--output", str(tmp_path / "eval.json"), "--allow-gold-replay-baseline"],
        [sys.executable, "training/run_unseen_db_benchmark.py", "--input", str(unseen), "--output", str(tmp_path / "unseen.json"), "--allow-gold-replay-baseline"],
        [sys.executable, "training/build_generic_ir_corpus.py", "--datasets", "missing-dataset", "--max-examples", "0", "--output-dir", str(tmp_path / "processed"), "--artifact-dir", str(tmp_path / "generic_artifacts")],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr or completed.stdout

    assert (tmp_path / "rag" / "example_index.pkl").exists()
    assert (tmp_path / "hard.jsonl").exists()
    assert (tmp_path / "eval.json").exists()
    assert (tmp_path / "unseen.md").exists()
    assert (tmp_path / "processed" / "generic_ir_train.jsonl").exists()
