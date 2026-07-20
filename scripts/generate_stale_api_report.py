"""Generate stale API report for NL2SQL repository.

Scans all Python files for nl2sql_v1 imports, v1 version strings,
deprecated aliases, and stale method calls. Classifies each occurrence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _classify_path(filepath: Path) -> str:
    """Classify a file path into usage domain."""
    rel = filepath.relative_to(ROOT).as_posix()
    if rel.startswith("tests/legacy/"):
        return "HISTORICAL_TEST"
    if rel.startswith("tests/"):
        return "ACTIVE_TEST"
    if rel.startswith("nl2sql_v1/"):
        return "LEGACY_MODULE"
    if rel.startswith("training_ir/"):
        return "MIGRATION_ONLY"
    if rel.startswith("app/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("inference/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("validation/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("retrieval/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("retriever/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("ir/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("neural_ir/"):
        return "ACTIVE_TRAINING"
    if rel.startswith("training/"):
        return "ACTIVE_TRAINING"
    if rel.startswith("dataset_training/"):
        return "ACTIVE_TRAINING"
    if rel.startswith("orchestration/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("model_bundle/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("generic_planner/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("semantic_layer/"):
        return "ACTIVE_RUNTIME"
    if rel.startswith("scripts/"):
        return "MIGRATION_ONLY"
    if rel.startswith("docs/"):
        return "HISTORICAL_DOC"
    return "UNKNOWN"


def scan_nl2sql_v1_imports() -> list[dict]:
    """Scan for all `from nl2sql_v1` import statements."""
    results = []
    for py_file in ROOT.rglob("*.py"):
        if "venv" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_num, line in enumerate(content.splitlines(), 1):
            if re.match(r"\s*from\s+nl2sql_v1", line):
                results.append({
                    "file": py_file.relative_to(ROOT).as_posix(),
                    "line": line_num,
                    "content": line.strip(),
                    "classification": _classify_path(py_file),
                    "issue_type": "nl2sql_v1_import",
                })
    return results


def scan_v1_version_strings() -> list[dict]:
    """Scan for model version strings containing v1."""
    results = []
    patterns = [
        r"schema_aware_queryir_v1",
        r"neural_queryir_v1",
        r"option_a_v2",
    ]
    for py_file in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.yaml")):
        if "venv" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    results.append({
                        "file": py_file.relative_to(ROOT).as_posix(),
                        "line": line_num,
                        "content": line.strip(),
                        "classification": _classify_path(py_file),
                        "issue_type": "v1_version_string",
                        "pattern": pattern,
                    })
    return results


def scan_stale_api_calls() -> list[dict]:
    """Scan for calls to APIs that no longer exist."""
    results = []
    stale_patterns = [
        ("check_leakage(", "stale_method_call"),
        ("_load_model_legacy", "deprecated_method"),
        ("option_a_model_dir", "deprecated_alias"),
        ("use_option_a_fallback", "deprecated_alias"),
        ("option_a_threshold", "deprecated_alias"),
    ]
    for py_file in ROOT.rglob("*.py"):
        if "venv" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern, issue in stale_patterns:
                if pattern in line and not line.strip().startswith("#"):
                    results.append({
                        "file": py_file.relative_to(ROOT).as_posix(),
                        "line": line_num,
                        "content": line.strip(),
                        "classification": _classify_path(py_file),
                        "issue_type": issue,
                        "pattern": pattern,
                    })
    return results


def main() -> None:
    v1_imports = scan_nl2sql_v1_imports()
    v1_versions = scan_v1_version_strings()
    stale_calls = scan_stale_api_calls()

    all_issues = v1_imports + v1_versions + stale_calls

    # Summary by classification
    by_classification: dict[str, int] = {}
    for issue in all_issues:
        cls = issue["classification"]
        by_classification[cls] = by_classification.get(cls, 0) + 1

    by_type: dict[str, int] = {}
    for issue in all_issues:
        t = issue["issue_type"]
        by_type[t] = by_type.get(t, 0) + 1

    report = {
        "total_issues": len(all_issues),
        "summary_by_classification": by_classification,
        "summary_by_type": by_type,
        "active_runtime_issues": [i for i in all_issues if i["classification"] == "ACTIVE_RUNTIME"],
        "active_training_issues": [i for i in all_issues if i["classification"] == "ACTIVE_TRAINING"],
        "active_test_issues": [i for i in all_issues if i["classification"] == "ACTIVE_TEST"],
        "historical_test_issues": [i for i in all_issues if i["classification"] == "HISTORICAL_TEST"],
        "migration_only_issues": [i for i in all_issues if i["classification"] == "MIGRATION_ONLY"],
        "all_issues": all_issues,
    }

    output_path = ROOT / "artifacts" / "architecture" / "stale_api_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Stale API report written to {output_path}")
    print(f"Total issues: {len(all_issues)}")
    print(f"Active runtime: {by_classification.get('ACTIVE_RUNTIME', 0)}")
    print(f"Active training: {by_classification.get('ACTIVE_TRAINING', 0)}")
    print(f"Active test: {by_classification.get('ACTIVE_TEST', 0)}")


if __name__ == "__main__":
    main()
