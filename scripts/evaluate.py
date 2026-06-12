from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nl2sql_v1.engine import NL2SQLEngine
from nl2sql_v1.executor import execute_select
from nl2sql_v1.retriever import TfidfRetriever, load_examples
from nl2sql_v1.schema import read_sqlite_schema
from nl2sql_v1.validator import validate_select_sql


def evaluate(db_path: Path) -> None:
    examples_path = ROOT / "training_data" / "examples.jsonl"
    model_path = ROOT / "models" / "tfidf_retriever.joblib"
    retriever = TfidfRetriever.load_or_train(model_path, examples_path)
    engine = NL2SQLEngine(
        retriever=retriever,
        templates_path=ROOT / "data" / "templates.yaml",
        synonyms_path=ROOT / "data" / "synonyms.yaml",
    )
    schema = read_sqlite_schema(db_path)
    examples = load_examples(examples_path)

    retrieval_hits = 0
    valid_sql = 0
    executable = 0
    for row in examples:
        result = engine.generate(row["question"], schema)
        if result.retrieval.example_id == row["id"]:
            retrieval_hits += 1
        validation = validate_select_sql(result.sql, schema)
        valid_sql += int(validation.ok)
        try:
            execute_select(db_path, result.sql)
            executable += 1
        except Exception:
            pass

    main = engine.generate("Top 5 customers by sales", schema)
    print(f"examples: {len(examples)}")
    print(f"retrieval_accuracy: {retrieval_hits / len(examples):.3f}")
    print(f"sql_validation_rate: {valid_sql / len(examples):.3f}")
    print(f"executable_rate: {executable / len(examples):.3f}")
    print("main_question_sql:")
    print(main.sql)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "sample_retail.db")
    args = parser.parse_args()
    evaluate(args.db)


if __name__ == "__main__":
    main()
