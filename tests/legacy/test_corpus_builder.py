from __future__ import annotations

from datasets.corpus_builder import CorpusBuilder
from datasets.models import DatabaseSchema, Text2SQLExample


class FakeLoader:
    def load(self, dataset_name: str, max_examples: int | None = None):
        schema = DatabaseSchema(
            db_id="shop",
            dataset_name="mock",
            tables={"orders": {"columns": ["customer_id", "amount"]}},
        )
        examples = [
            Text2SQLExample(
                example_id="supported",
                dataset_name="mock",
                db_id="shop",
                question="sales by customer",
                sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
                split="train",
            ),
            Text2SQLExample(
                example_id="unsupported",
                dataset_name="mock",
                db_id="shop",
                question="above average orders",
                sql="SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)",
                split="train",
            ),
        ]
        return examples, {"shop": schema}


def test_corpus_builder_splits_supported_and_unsupported(tmp_path):
    builder = CorpusBuilder(loader=FakeLoader(), output_dir=tmp_path)
    payload = builder.build_corpus(["mock"])
    builder.save_outputs(payload, output_dir=tmp_path)

    assert payload["stats"].supported_examples == 1
    assert payload["stats"].unsupported_examples == 1
    assert (tmp_path / "supported_examples.jsonl").exists()
    assert (tmp_path / "unsupported_examples.jsonl").exists()
