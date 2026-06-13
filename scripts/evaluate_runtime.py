from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_golden_tests import DEFAULT_ARTIFACT_DIR, DEFAULT_DB, DEFAULT_GOLDEN_FILE, run_golden_tests


DEFAULT_OUTPUT = ROOT / "evaluation" / "runtime_evaluation_report.json"


def evaluate_runtime(
    db_path: Path = DEFAULT_DB,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    golden_file: Path = DEFAULT_GOLDEN_FILE,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    return run_golden_tests(
        db_path=db_path,
        artifact_dir=artifact_dir,
        golden_file=golden_file,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the canonical QueryIR runtime.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--golden-file", type=Path, default=DEFAULT_GOLDEN_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = evaluate_runtime(args.db, args.artifact_dir, args.golden_file, args.output)
    print(json.dumps({key: payload[key] for key in payload if key != "case_results"}, indent=2))
    return 0 if payload["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
