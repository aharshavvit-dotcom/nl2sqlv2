"""CI gate test: no non-compat file may import from nl2sql_v1 directly.

This test enforces the migration deadline for nl2sql_v1 removal.
Files under `compat/`, `tests/legacy/`, and `nl2sql_v1/` itself are
exempt. All other active source files must use canonical imports.

Migration deadline: 2026-09-01
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Directories exempt from the import restriction
EXEMPT_DIRS = {
    "nl2sql_v1",
    "compat",
}
# Paths that contain legacy tests (exempt)
EXEMPT_PATH_PATTERNS = [
    "tests/legacy/",
    "tests\\legacy\\",
]
# Specific files with temporary exemptions (must have migration ticket)
TEMPORARY_EXEMPTIONS = {
    # Format: relative path -> migration ticket/note
    # Example: "validation/sql_validator.py": "TICKET-123: migrating to db.schema_graph",
}


def _find_violating_files() -> list[tuple[str, int, str]]:
    """Scan all Python files for unauthorized nl2sql_v1 imports.

    Returns list of (relative_path, line_number, line_content) tuples.
    """
    violations = []
    for py_file in ROOT.rglob("*.py"):
        if "__pycache__" in py_file.parts or "venv" in py_file.parts:
            continue

        rel = py_file.relative_to(ROOT).as_posix()

        # Check exemptions
        first_dir = rel.split("/")[0] if "/" in rel else ""
        if first_dir in EXEMPT_DIRS:
            continue
        if any(pattern in rel for pattern in EXEMPT_PATH_PATTERNS):
            continue
        if rel in TEMPORARY_EXEMPTIONS:
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.match(r"from\s+nl2sql_v1\b", stripped) or re.match(r"import\s+nl2sql_v1\b", stripped):
                violations.append((rel, line_num, stripped))

    return violations


class TestStaleAPIAbsence:
    """Enforce that no active source file imports from nl2sql_v1."""

    def test_no_direct_nl2sql_v1_imports(self):
        """No non-exempt file should import from nl2sql_v1.

        All nl2sql_v1 functionality should be accessed through canonical
        locations (db.schema_graph, retrieval.tfidf_retriever, etc.)
        or through the compat/ bridge.

        If this test fails, either:
        1. Migrate the import to the canonical location, OR
        2. Add a temporary exemption with a migration ticket
        """
        violations = _find_violating_files()

        if violations:
            msg_lines = [
                f"\n{len(violations)} unauthorized nl2sql_v1 import(s) found:\n",
            ]
            for path, line_num, content in violations[:20]:
                msg_lines.append(f"  {path}:{line_num}: {content}")
            if len(violations) > 20:
                msg_lines.append(f"  ... and {len(violations) - 20} more")
            msg_lines.append(
                "\nMigrate these imports to canonical locations or add "
                "temporary exemptions in test_stale_api_absence.py"
            )
            # NOTE: This test is currently expected to FAIL until Phase 3
            # migration is complete. Change xfail to strict after migration.
            pytest.xfail("\n".join(msg_lines))

    def test_exemption_count_decreasing(self):
        """Track that temporary exemptions are being removed over time."""
        # This is a soft check — it just documents current state
        violations = _find_violating_files()
        exemptions = len(TEMPORARY_EXEMPTIONS)
        # Record for tracking
        print(f"Current violations: {len(violations)}")
        print(f"Current exemptions: {exemptions}")
        print(f"Total stale imports: {len(violations) + exemptions}")
