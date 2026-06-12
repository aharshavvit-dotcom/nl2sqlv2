from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nl2sql_v1.retriever import TfidfRetriever
from scripts.dataset_paths import ARTIFACT_DIR
from training.train_retriever_from_datasets import _atomic_json


def evaluate_retriever(
    artifact_dir: Path = ARTIFACT_DIR,
    splits: list[str] | None = None,
) -> dict[str, Any]:
    retriever = TfidfRetriever.load(artifact_dir)
    examples = _load_eval_examples(artifact_dir, splits=splits or ["validation", "test"])
    if not examples:
        examples = retriever.examples
    if not examples:
        raise ValueError(f"No training examples found in {artifact_dir}")

    top_1 = 0
    top_3 = 0
    top_5 = 0
    intent_hits = 0
    dataset_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "top_1": 0, "top_5": 0})
    template_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "top_1": 0, "top_5": 0})

    for example in examples:
        expected_template = example.get("template_id")
        expected_intent = example.get("intent")
        results = retriever.query(example["question"], top_k=5)
        templates = [result.template_id for result in results]
        intents = [result.example.get("intent") for result in results]
        dataset_name = example.get("dataset_name", "unknown")
        template_name = expected_template or "unknown"
        dataset_breakdown[dataset_name]["total"] += 1
        template_breakdown[template_name]["total"] += 1
        if templates and templates[0] == expected_template:
            top_1 += 1
            dataset_breakdown[dataset_name]["top_1"] += 1
            template_breakdown[template_name]["top_1"] += 1
        if expected_template in templates[:3]:
            top_3 += 1
        if expected_template in templates[:5]:
            top_5 += 1
            dataset_breakdown[dataset_name]["top_5"] += 1
            template_breakdown[template_name]["top_5"] += 1
        if expected_intent and intents and intents[0] == expected_intent:
            intent_hits += 1

    total = len(examples)
    report = {
        "example_count": total,
        "evaluation_splits": sorted({row.get("split", "unknown") for row in examples}),
        "top_1_template_accuracy": top_1 / total,
        "top_3_template_accuracy": top_3 / total,
        "top_5_template_accuracy": top_5 / total,
        "intent_accuracy": intent_hits / total,
        "dataset_breakdown": _with_accuracy(dataset_breakdown),
        "template_breakdown": _with_accuracy(template_breakdown),
    }
    _atomic_json(artifact_dir / "evaluation_report.json", report)
    return report


def _load_eval_examples(artifact_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for split in splits:
        path = artifact_dir / f"{split}_examples.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    return examples


def _with_accuracy(raw: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    payload: dict[str, dict[str, float | int]] = {}
    for key, value in raw.items():
        total = value["total"]
        payload[key] = {
            **value,
            "top_1_template_accuracy": value["top_1"] / total if total else 0.0,
            "top_5_template_accuracy": value["top_5"] / total if total else 0.0,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--splits", default="validation,test")
    args = parser.parse_args()
    report = evaluate_retriever(args.artifact_dir, splits=[item.strip() for item in args.splits.split(",") if item.strip()])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
