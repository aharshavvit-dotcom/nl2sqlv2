"""Run the Adaptive QueryIR Router benchmark.

This is the canonical entry point for benchmarking the router.
It wraps ``training_ir.benchmark_hybrid_system``.

Usage:
    python evaluation/run_adaptive_router_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from training_ir.benchmark_hybrid_system import main as benchmark_main
    benchmark_main()


if __name__ == "__main__":
    main()
