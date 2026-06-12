from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nl2sql_v1.retriever import TfidfRetriever

EXAMPLES = ROOT / "training_data" / "examples.jsonl"
MODEL = ROOT / "models" / "tfidf_retriever.joblib"


def main() -> None:
    retriever = TfidfRetriever.train(EXAMPLES)
    retriever.save(MODEL)
    print(f"Saved retriever with {len(retriever.examples)} examples to {MODEL}")


if __name__ == "__main__":
    main()
