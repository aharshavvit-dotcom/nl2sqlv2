from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.corpus_builder import CorpusBuilder
from scripts.dataset_paths import ARTIFACT_DIR, PROCESSED_DATA_DIR, ensure_dataset_dirs, parse_dataset_list
from training.training_report import build_training_report
from retrieval.artifact_compatibility import build_sklearn_metadata, write_sklearn_metadata


def train_from_datasets(
    dataset_names: list[str],
    artifact_dir: Path = ARTIFACT_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    max_examples: int | None = None,
    include_schema_text: bool = False,
    train_splits: list[str] | None = None,
) -> dict[str, Any]:
    ensure_dataset_dirs()
    requested = parse_dataset_list(dataset_names)
    if requested == ["handwritten"]:
        return _train_from_handwritten_examples(
            artifact_dir=artifact_dir,
            train_splits=train_splits or ["train"],
        )
    builder = CorpusBuilder(output_dir=output_dir)
    payload = builder.build_corpus(
        requested,
        max_examples=max_examples,
        include_schema_text=include_schema_text,
    )
    builder.save_outputs(payload, output_dir=output_dir)
    supported = [example for example in payload["examples"] if example.is_supported]
    if not supported:
        raise ValueError("No supported examples found. Verify datasets and classifier coverage.")

    train_split_set = set(train_splits or ["train"])
    train_examples = [example for example in supported if example.split in train_split_set]
    used_fallback_all_splits = False
    if not train_examples:
        train_examples = supported
        used_fallback_all_splits = True

    training_examples = [
        _training_row(example, include_schema_text=include_schema_text)
        for example in train_examples
    ]
    validation_examples = [
        _training_row(example, include_schema_text=include_schema_text)
        for example in supported
        if example.split == "validation"
    ]
    test_examples = [
        _training_row(example, include_schema_text=include_schema_text)
        for example in supported
        if example.split == "test"
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
    matrix = vectorizer.fit_transform([row["training_text"] for row in training_examples])
    report = build_training_report(
        datasets_used=requested,
        total_loaded=payload["stats"].total_examples,
        supported=len(training_examples),
        unsupported=payload["stats"].unsupported_examples,
        examples=[_training_row(example, include_schema_text=include_schema_text) for example in supported],
        vocabulary_size=len(vectorizer.vocabulary_),
        include_schema_text=include_schema_text,
    )
    report["supported_examples_all_splits"] = payload["stats"].supported_examples
    report["train_examples"] = len(training_examples)
    report["validation_examples"] = len(validation_examples)
    report["test_examples"] = len(test_examples)
    report["train_splits"] = sorted(train_split_set)
    report["used_fallback_all_splits"] = used_fallback_all_splits
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_joblib(artifact_dir / "tfidf_vectorizer.pkl", vectorizer)
    _atomic_joblib(artifact_dir / "tfidf_matrix.pkl", matrix)
    _atomic_jsonl(artifact_dir / "training_examples.jsonl", training_examples)
    _atomic_jsonl(artifact_dir / "train_examples.jsonl", training_examples)
    _atomic_jsonl(artifact_dir / "validation_examples.jsonl", validation_examples)
    _atomic_jsonl(artifact_dir / "test_examples.jsonl", test_examples)
    _atomic_json(artifact_dir / "supported_patterns.json", payload["stats"].by_template)
    _atomic_json(artifact_dir / "dataset_stats.json", payload["stats"].to_dict())
    _atomic_json(artifact_dir / "training_report.json", report)
    write_sklearn_metadata(
        artifact_dir,
        build_sklearn_metadata(
            artifact_types=["tfidf_vectorizer", "tfidf_matrix"],
            config={"datasets": requested, "include_schema_text": include_schema_text},
        ),
    )
    return {
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "training_report": report,
        "dataset_stats": payload["stats"].to_dict(),
    }


def _train_from_handwritten_examples(
    artifact_dir: Path,
    train_splits: list[str],
) -> dict[str, Any]:
    examples_path = ROOT / "training_data" / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Hand-written examples file not found: {examples_path}")

    rows: list[dict[str, Any]] = []
    with examples_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No hand-written examples found in {examples_path}")

    split_set = set(train_splits)
    training_examples = [row for row in rows if row.get("split") in split_set]
    validation_examples = [row for row in rows if row.get("split") == "validation"]
    test_examples = [row for row in rows if row.get("split") == "test"]
    if not training_examples:
        raise ValueError("No hand-written training examples found for the requested train splits.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
    matrix = vectorizer.fit_transform([row["training_text"] for row in training_examples])
    report = build_training_report(
        datasets_used=["handwritten"],
        total_loaded=len(rows),
        supported=len(training_examples),
        unsupported=0,
        examples=rows,
        vocabulary_size=len(vectorizer.vocabulary_),
        include_schema_text=False,
    )
    report["supported_examples_all_splits"] = len(rows)
    report["train_examples"] = len(training_examples)
    report["validation_examples"] = len(validation_examples)
    report["test_examples"] = len(test_examples)
    report["train_splits"] = sorted(split_set)
    report["used_fallback_all_splits"] = False
    dataset_stats = {
        "total_examples": len(rows),
        "supported_examples": len(rows),
        "unsupported_examples": 0,
        "by_dataset": {"handwritten": len(rows)},
        "by_template": report["by_template"],
        "by_split": {
            split: sum(1 for row in rows if row.get("split") == split)
            for split in sorted({str(row.get("split")) for row in rows})
        },
        "unsupported_reasons": {},
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_joblib(artifact_dir / "tfidf_vectorizer.pkl", vectorizer)
    _atomic_joblib(artifact_dir / "tfidf_matrix.pkl", matrix)
    _atomic_jsonl(artifact_dir / "training_examples.jsonl", training_examples)
    _atomic_jsonl(artifact_dir / "train_examples.jsonl", training_examples)
    _atomic_jsonl(artifact_dir / "validation_examples.jsonl", validation_examples)
    _atomic_jsonl(artifact_dir / "test_examples.jsonl", test_examples)
    _atomic_json(artifact_dir / "supported_patterns.json", report["by_template"])
    _atomic_json(artifact_dir / "dataset_stats.json", dataset_stats)
    _atomic_json(artifact_dir / "training_report.json", report)
    write_sklearn_metadata(
        artifact_dir,
        build_sklearn_metadata(
            artifact_types=["tfidf_vectorizer", "tfidf_matrix"],
            source_path=examples_path,
            config={"datasets": ["handwritten"]},
        ),
    )
    return {
        "artifact_dir": str(artifact_dir),
        "output_dir": str(PROCESSED_DATA_DIR),
        "training_report": report,
        "dataset_stats": dataset_stats,
    }


def _training_row(example: Any, include_schema_text: bool) -> dict[str, Any]:
    serialized_schema = None
    if example.schema:
        serialized_schema = example.schema.get("serialized_schema")
    serialized_schema = serialized_schema or example.extracted_slots.get("serialized_schema")
    training_text = example.question
    if include_schema_text and serialized_schema:
        training_text = f"{example.question} {serialized_schema}"
    return {
        "id": example.example_id,
        "example_id": example.example_id,
        "dataset_name": example.dataset_name,
        "db_id": example.db_id,
        "question": example.question,
        "training_text": training_text,
        "sql": example.sql,
        "split": example.split,
        "template_id": example.template_id,
        "intent": example.intent,
        "metric": example.extracted_slots.get("metric"),
        "dimension": example.extracted_slots.get("dimension"),
        "limit": example.extracted_slots.get("limit") or 10,
        "order": "DESC",
        "is_supported": example.is_supported,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _atomic_joblib(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, tmp)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--include-schema-text", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--train-splits", default="train")
    args = parser.parse_args()
    report = train_from_datasets(
        parse_dataset_list(args.datasets),
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        include_schema_text=args.include_schema_text,
        train_splits=parse_dataset_list(args.train_splits),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
