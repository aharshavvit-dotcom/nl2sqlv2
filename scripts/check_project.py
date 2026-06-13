from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPILE_TARGETS = [
    "app",
    "datasets",
    "execution",
    "inference",
    "ir",
    "nl2sql_v1",
    "retriever",
    "scripts",
    "tests",
    "training",
    "validation",
]


def run(command: list[str]) -> int:
    print("+ " + " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


def main() -> int:
    commands = [
        [sys.executable, "-m", "compileall", *COMPILE_TARGETS],
        [sys.executable, "-m", "pytest", "tests/"],
    ]
    for command in commands:
        code = run(command)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
