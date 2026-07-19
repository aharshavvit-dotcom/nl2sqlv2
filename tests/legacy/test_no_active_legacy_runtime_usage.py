"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATHS = [
    ROOT / "app",
    ROOT / "execution",
    ROOT / "inference",
    ROOT / "ir",
    ROOT / "retriever",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "validation",
    ROOT / "README.md",
]


def test_active_runtime_does_not_import_legacy_sql_engine_executor_or_validator() -> None:
    forbidden = [
        "NL2SQL" + "Engine",
        "nl2sql_v1." + "engine",
        "nl2sql_v1." + "executor",
        "nl2sql_v1." + "validator",
    ]
    offenders: list[str] = []

    for root in ACTIVE_PATHS:
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.suffix in {".py", ".md"}]
        for path in paths:
            if "tests/legacy" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT)} contains {token}")

    assert offenders == []

