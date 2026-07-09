"""Freeze benchmark splits and perform cross-split data leakage audits.

This implements Stage 0.5 (Create frozen benchmark snapshot and cross-split near-duplicate check).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_question(q: str) -> str:
    """Normalize question string for matching."""
    return " ".join(str(q or "").lower().strip().split())


def normalize_query_ir(ir: dict[str, Any] | None) -> str:
    """Normalize QueryIR structure to a sorted string representation."""
    if not ir:
        return ""
    # Strip volatile fields if any (e.g. metadata)
    cleaned = {k: v for k, v in ir.items() if k not in {"metadata", "confidence"}}
    return json.dumps(cleaned, sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except Exception:
                    pass
    return records


def freeze_benchmark() -> None:
    """Freeze test datasets and perform near-duplicate/leakage checks."""
    source_dir = ROOT / "data" / "processed"
    target_dir = ROOT / "artifacts" / "benchmarks" / "semantic_baseline_20260708"
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_freeze = [
        "generic_ir_test.jsonl",
        "generic_ir_unseen_db_test.jsonl",
        "generic_ir_validation.jsonl",
    ]

    print("=== Freezing Benchmark Splits ===")
    for filename in files_to_freeze:
        src = source_dir / filename
        dst = target_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {filename} to frozen benchmarks directory.")
        else:
            print(f"Warning: Source file {src} not found!")

    # Perform leakage/near-duplicate detection
    train_path = source_dir / "generic_ir_train.jsonl"
    print(f"\nLoading training data from {train_path} for leakage audit...")
    train_records = read_jsonl(train_path)
    print(f"Loaded {len(train_records)} training records.")

    train_questions = set()
    train_queries = set()
    for row in train_records:
        q_norm = normalize_question(row.get("question") or "")
        if q_norm:
            train_questions.add(q_norm)
        ir_norm = normalize_query_ir(row.get("query_ir"))
        if ir_norm:
            train_queries.add(ir_norm)

    leakage_report: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "benchmark_id": "semantic_baseline_20260708",
        "created_at": "2026-07-09T20:00:00Z",
        "files": {},
    }

    for filename in files_to_freeze:
        frozen_path = target_dir / filename
        if not frozen_path.exists():
            continue
        
        records = read_jsonl(frozen_path)
        sha256 = compute_file_sha256(frozen_path)
        metadata["files"][filename] = {
            "row_count": len(records),
            "sha256": sha256,
            "size_bytes": frozen_path.stat().st_size,
        }

        question_leaks = []
        query_leaks = []
        for row in records:
            ex_id = row.get("example_id")
            q = row.get("question") or ""
            q_norm = normalize_question(q)
            ir_norm = normalize_query_ir(row.get("query_ir"))

            is_q_leak = q_norm in train_questions
            is_ir_leak = ir_norm in train_queries

            if is_q_leak:
                question_leaks.append({
                    "example_id": ex_id,
                    "question": q,
                })
            if is_ir_leak:
                query_leaks.append({
                    "example_id": ex_id,
                    "query_ir": row.get("query_ir"),
                })

        leakage_report[filename] = {
            "total_examples": len(records),
            "question_duplicate_count": len(question_leaks),
            "query_duplicate_count": len(query_leaks),
            "question_duplicates": question_leaks[:50],  # cap list for reporting
            "query_duplicates": query_leaks[:50],
        }
        print(f"\nResults for {filename}:")
        print(f"  Total examples: {len(records)}")
        print(f"  Exact question matches in train: {len(question_leaks)}")
        print(f"  Exact QueryIR matches in train: {len(query_leaks)}")

    # Write outputs
    (target_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (target_dir / "benchmark_leakage_report.json").write_text(
        json.dumps(leakage_report, indent=2), encoding="utf-8"
    )
    print("\nBenchmark freeze complete! Files and reports saved in:")
    print(f"  {target_dir}")


if __name__ == "__main__":
    freeze_benchmark()
