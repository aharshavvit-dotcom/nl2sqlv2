"""Run model evaluation against golden test cases.

This is the canonical entry point for model evaluation.

Usage:
    python evaluation/run_model_evaluation.py [--db-path data/sample_retail.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--db-path", type=str, default=str(ROOT / "data" / "sample_retail.db"))
    args = parser.parse_args()

    from scripts.run_golden_tests import main as golden_main
    golden_main()


if __name__ == "__main__":
    main()
