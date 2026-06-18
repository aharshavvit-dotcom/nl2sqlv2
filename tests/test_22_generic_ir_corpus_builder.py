from __future__ import annotations

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
