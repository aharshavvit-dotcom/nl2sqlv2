"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets.wikisql_adapter import WikiSQLAdapter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_wikisql_adapter_converts_structured_sql(tmp_path: Path) -> None:
    raw = tmp_path / "wikisql"
    raw.mkdir()
    table = {"id": "table_1", "header": ["Year", "Revenue"]}
    row = {"table_id": "table_1", "question": "revenue in 2020", "sql": {"sel": 1, "agg": 4, "conds": [[0, 0, "2020"]]}}
    for split in ["train", "dev", "test"]:
        _write_jsonl(raw / f"{split}.tables.jsonl", [table])
        _write_jsonl(raw / f"{split}.jsonl", [row])

    adapter = WikiSQLAdapter()
    sql = adapter.convert_wikisql_to_sql(row, table)
    examples = adapter.load_examples(raw)

    assert 'SELECT SUM("Revenue")' in sql
    assert '"Year" = ' in sql
    assert len(examples) == 3
    assert examples[0].difficulty == "easy"
    assert [example.split for example in examples] == ["train", "validation", "test"]
