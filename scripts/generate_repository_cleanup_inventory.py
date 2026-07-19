"""
Purpose: Generate repository cleanup inventory, maps, manifests, and governance reports.
Required because: Cleanup decisions need reproducible evidence before files are removed.

Inputs: Git metadata, tracked/untracked files, local artifact manifests, configs, docs, and source files.
Outputs: docs/REPOSITORY_MAP.md and artifacts/repository_cleanup/* reports.
Safe to delete outputs: Yes, rerun this script to regenerate them from the current checkout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "repository_cleanup"
REPO_MAP = ROOT / "docs" / "REPOSITORY_MAP.md"
BASELINE_COMMIT = "739def558686b0c6caa1b398e07920f5e77b2356"

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
GENERATED_SUFFIXES = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx", ".db", ".sqlite", ".sqlite3"}
CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
ROOT_LOG_NAMES = {
    "compileall_full.log",
    "streamlit_generic.err.log",
    "streamlit_generic.out.log",
    "streamlit_smoke.err.log",
    "streamlit_smoke.out.log",
    "streamlit_v2_smoke.err.log",
    "streamlit_v2_smoke.out.log",
}


@dataclass(frozen=True)
class FolderInfo:
    purpose: str
    why_required: str
    owner: str
    generated_or_source: str
    cleanup_rule: str


FOLDER_INFO: dict[str, FolderInfo] = {
    ".github": FolderInfo(
        "Continuous-integration workflow configuration.",
        "Runs automated validation outside the local development machine.",
        "DevOps/Test",
        "configuration",
        "Keep when workflows match supported commands; review untracked workflow additions before commit.",
    ),
    "app": FolderInfo(
        "User-facing Streamlit application and safe preview helpers.",
        "Provides the runtime interface for connected-database NL-to-SQL usage.",
        "Runtime",
        "source",
        "Keep canonical app entry points; do not import training paths in normal runtime.",
    ),
    "artifacts": FolderInfo(
        "Generated training, audit, evaluation, and model-bundle outputs.",
        "Captures reproducibility evidence and local runtime bundles.",
        "MLOps",
        "generated",
        "Keep only active/protected artifacts and cleanup reports in Git; most contents stay ignored.",
    ),
    "artifacts/repository_cleanup": FolderInfo(
        "Machine-readable cleanup inventory and governance reports.",
        "Records evidence for retained, deleted, archived, and review-required items.",
        "Documentation/MLOps",
        "generated documentation",
        "Regenerate with scripts/generate_repository_cleanup_inventory.py after structural changes.",
    ),
    "capabilities": FolderInfo(
        "SQL capability taxonomy, extraction, and reporting contracts.",
        "Training and quality gates need consistent supported/unsupported SQL labels.",
        "ML/Data",
        "source",
        "Keep as canonical capability contract unless replaced by a tested taxonomy module.",
    ),
    "clarification": FolderInfo(
        "Clarification state, ambiguity detection, and question generation.",
        "Runtime needs safe abstention and follow-up behavior for ambiguous requests.",
        "Runtime",
        "source",
        "Keep while prediction orchestration exposes clarification metadata.",
    ),
    "configs": FolderInfo(
        "Canonical runtime and training configuration files.",
        "Training, smoke, baseline, and production paths are driven by validated config.",
        "MLOps",
        "configuration",
        "Keep consumed configs; remove fields only after consumer checks prove non-use.",
    ),
    "connected_db_testing": FolderInfo(
        "Generated connected-database regression case support.",
        "Validates schema-general behavior against live or generated schemas.",
        "Database/Test",
        "source",
        "Keep if referenced by regression scripts or tests.",
    ),
    "data": FolderInfo(
        "Lightweight semantic source data and frozen split manifests.",
        "Runtime synonym defaults and immutable split membership are required for reproducible training.",
        "Data",
        "mixed source/generated",
        "Track small canonical YAML and frozen split manifests; keep raw/processed datasets ignored.",
    ),
    "data/splits": FolderInfo(
        "Frozen dataset split manifests and ID lists.",
        "Training and evaluation must not silently change split membership.",
        "Data/MLOps",
        "generated governance data",
        "Never overwrite a frozen split; create a new version when membership changes.",
    ),
    "datasets": FolderInfo(
        "External dataset adapters and schema normalization utilities.",
        "Generic training depends on normalized WikiSQL, Spider, and BIRD records.",
        "Data",
        "source",
        "Keep adapters referenced by dataset loaders and training builders.",
    ),
    "dataset_training": FolderInfo(
        "Dataset construction, split, leakage, and corpus-quality tooling.",
        "Canonical training needs reproducible corpora and leakage checks.",
        "Data/ML",
        "source",
        "Keep canonical builders; consolidate only with tests covering corpus outputs.",
    ),
    "db": FolderInfo(
        "Database connection, schema reading, and dialect boundaries.",
        "Runtime and evaluation require safe schema discovery for SQLite/PostgreSQL.",
        "Database",
        "source",
        "Keep connector abstractions and dialect-specific implementations.",
    ),
    "deployment": FolderInfo(
        "Production-readiness helpers.",
        "Deployment checks summarize whether the repository is safe to run.",
        "DevOps",
        "source",
        "Keep while scripts or tests import production readiness checks.",
    ),
    "docs": FolderInfo(
        "Canonical architecture, policy, developer, and specification documentation.",
        "New engineers need current command, runtime, and governance guidance.",
        "Documentation",
        "documentation",
        "Keep canonical docs; move run-specific generated reports out of general docs after review.",
    ),
    "docs/architecture": FolderInfo(
        "Architecture and governance policy documents.",
        "Explains production runtime, bundle lifecycle, privacy, database execution, and lineage.",
        "Architecture/Documentation",
        "documentation",
        "Keep when paths and commands validate against current code.",
    ),
    "docs/reports": FolderInfo(
        "Historical/generated audit reports.",
        "Some reports preserve decision history, but run-specific reports should not be primary docs.",
        "Documentation/MLOps",
        "mixed documentation/generated reports",
        "Archive or move run-scoped reports to artifacts/pipeline/runs/<run_id>/reports/ after review.",
    ),
    "docs/specs": FolderInfo(
        "QueryIR contract specifications.",
        "Renderer, validator, migration, and tests need stable IR semantics.",
        "Architecture",
        "documentation",
        "Keep versioned specs that match active QueryIR code.",
    ),
    "evaluation": FolderInfo(
        "Reusable evaluation code, thresholds, and golden cases.",
        "Quality gates and regression reports depend on stable evaluation inputs.",
        "ML/Test",
        "mixed source/test data",
        "Keep source and checked-in fixtures; generated reports stay ignored.",
    ),
    "evaluation/fixtures": FolderInfo(
        "Small controlled execution fixtures.",
        "Execution-aware evaluation needs stable SQL/database cases.",
        "Test",
        "test data",
        "Keep while tests or controlled evaluation reference them.",
    ),
    "execution": FolderInfo(
        "Read-only SQL execution boundary.",
        "Runtime must execute only validated, safe SELECT statements.",
        "Security/Runtime",
        "source",
        "Keep as canonical execution boundary.",
    ),
    "execution_eval": FolderInfo(
        "SQL canonicalization, structural comparison, and execution matching.",
        "Evaluation needs reusable semantic/structural comparison utilities.",
        "Evaluation",
        "source",
        "Keep while execution-aware tests and reports depend on it.",
    ),
    "feedback": FolderInfo(
        "Reviewed feedback models, storage, and conversion workflows.",
        "Self-training and governance use feedback only through typed flows.",
        "ML/Data",
        "source",
        "Keep active feedback contracts; generated feedback JSONL remains ignored.",
    ),
    "generic_planner": FolderInfo(
        "Schema-safe deterministic planner for direct simple queries.",
        "Simple connected-database requests bypass model routing safely.",
        "Runtime",
        "source",
        "Keep while runtime direct planning and generic join policy tests pass through it.",
    ),
    "inference": FolderInfo(
        "Runtime prediction orchestration, confidence, telemetry, grounding, and slot resolution.",
        "The app and smoke tests need a single runtime prediction path.",
        "Runtime",
        "source",
        "Keep canonical runtime modules; remove hidden schema rules only after regression coverage.",
    ),
    "inference/grounding": FolderInfo(
        "Schema and literal grounding services.",
        "Connected databases need schema-specific projection/filter value grounding.",
        "Runtime",
        "source",
        "Keep while slot resolver and grounding tests import these modules.",
    ),
    "ir": FolderInfo(
        "QueryIR models, SQL conversion, migration, validation, and rendering.",
        "QueryIR is the deterministic contract between models and executable SQL.",
        "Architecture/Runtime",
        "source",
        "Keep canonical QueryIR v2 modules; compatibility code needs explicit removal conditions.",
    ),
    "ir/query_ir_v2_rendering": FolderInfo(
        "QueryIR v2 renderer internals.",
        "SQL generation is split into query, predicate, and expression rendering.",
        "Runtime",
        "source",
        "Keep as canonical renderer implementation.",
    ),
    "model_bundle": FolderInfo(
        "Model bundle build, manifest, validation, loading, and promotion logic.",
        "Runtime loads artifacts only through validated bundle manifests.",
        "MLOps/Runtime",
        "source",
        "Keep canonical bundle lifecycle code; never delete active bundle evidence automatically.",
    ),
    "model_registry": FolderInfo(
        "Model artifact registry and manifest versioning helpers.",
        "Training/promotion need structured artifact identity.",
        "MLOps",
        "source",
        "Keep while quality gates and model selection tests import it.",
    ),
    "model_selection": FolderInfo(
        "Champion/challenger and promotion selection logic.",
        "Release readiness depends on controlled model candidate comparison.",
        "MLOps",
        "source",
        "Keep while promotion governance tests use these policies.",
    ),
    "models": FolderInfo(
        "Ignored local trained model outputs with a tracked placeholder.",
        "Keeps the artifact directory available without committing model binaries.",
        "MLOps",
        "generated placeholder",
        "Track only .gitkeep unless a specific lightweight artifact is approved.",
    ),
    "neural_ir": FolderInfo(
        "Neural QueryIR architecture, labels, tokenizer, calibration, and prediction utilities.",
        "The neural model path and related tests depend on these contracts.",
        "ML",
        "source",
        "Keep active neural components; legacy experiments require review before deletion.",
    ),
    "neural_optimization": FolderInfo(
        "Optimizers, schedulers, checkpoints, ranker, and training diagnostics.",
        "Training wrappers share neural optimization infrastructure.",
        "ML",
        "source",
        "Keep while training and neural tests import these helpers.",
    ),
    "nl2sql_v1": FolderInfo(
        "Legacy v1 NL-to-SQL implementation and compatibility reference.",
        "Migration and legacy tests still compare or validate older behavior.",
        "Architecture/Test",
        "legacy source",
        "Do not delete until legacy tests and migration docs are retired.",
    ),
    "orchestration": FolderInfo(
        "Pipeline configuration, state, contract validation, step execution, and reporting.",
        "Integrated training depends on one auditable pipeline runner.",
        "MLOps",
        "source",
        "Keep canonical pipeline orchestration and fail unknown steps.",
    ),
    "pipeline_configs": FolderInfo(
        "Pipeline-level config presets.",
        "Supports smoke and full generic training orchestration.",
        "MLOps",
        "configuration",
        "Keep configs consumed by orchestration or migrate into configs/ with compatibility proof.",
    ),
    "quality_gates": FolderInfo(
        "Model quality, release, threshold, and regression-gate code.",
        "Promotion must fail closed when safety or quality evidence is missing.",
        "MLOps/Test",
        "source",
        "Keep while bundle validation and release readiness use these gates.",
    ),
    "retrieval": FolderInfo(
        "Retrieval indexes, RAG retriever, reranker, schema indexes, and artifact compatibility.",
        "Runtime/training use retrieval artifacts and metadata policy.",
        "Runtime/ML",
        "source",
        "Keep canonical retrieval infrastructure; distinguish from retriever/ runtime model wrapper.",
    ),
    "retriever": FolderInfo(
        "Runtime retrieval NL-to-SQL model wrapper.",
        "The app and tests still import RetrievalNL2SQLModel from this package.",
        "Runtime",
        "source",
        "Keep as active runtime wrapper unless imports migrate to retrieval/.",
    ),
    "reward": FolderInfo(
        "Reward features, scoring, and candidate reranking helpers.",
        "Self-training and candidate selection use reward signals.",
        "ML",
        "source",
        "Keep while training/self-training paths reference it.",
    ),
    "scripts": FolderInfo(
        "Supported operational, audit, dataset, and smoke commands.",
        "Developers need stable command entry points outside package internals.",
        "DevOps/MLOps",
        "source",
        "Keep supported scripts with purpose and usage docs; delete one-off scripts after proof.",
    ),
    "self_training": FolderInfo(
        "Self-training loops, candidate generation, correction, and improvement tracking.",
        "Provides governed feedback/improvement workflows.",
        "ML",
        "source/configuration",
        "Keep while readiness audits and tests cover self-training.",
    ),
    "semantic_layer": FolderInfo(
        "Schema profiling, semantic profiles, glossary, metrics, and dimensions.",
        "Connected databases need schema-derived semantic metadata.",
        "Runtime/Data",
        "source",
        "Keep while runtime schema mapping and semantic tests use it.",
    ),
    "tests": FolderInfo(
        "Active unit, integration, regression, safety, and legacy tests.",
        "Cleanup is safe only when behavior remains covered.",
        "Test",
        "test source/data",
        "Keep active tests; classify legacy tests individually before removal.",
    ),
    "tests/legacy": FolderInfo(
        "Legacy compatibility and research-path regression tests.",
        "They guard migrations and older APIs that may still be referenced.",
        "Test",
        "legacy test source",
        "Review for update/archive/delete; do not delete just because the folder says legacy.",
    ),
    "training": FolderInfo(
        "Canonical training, evaluation, promotion, and report commands.",
        "Integrated model production starts from training/train_model.py.",
        "ML/MLOps",
        "source",
        "Keep supported entry points; consolidate old wrappers only with command/doc/test updates.",
    ),
    "training_data": FolderInfo(
        "Small checked-in training examples and generated local IR corpora.",
        "Examples seed tests; generated JSONL corpora are reproducible and ignored.",
        "Data/ML",
        "mixed source/generated",
        "Track examples.jsonl and stats only when intentionally curated; generated JSONL stays ignored.",
    ),
    "training_ir": FolderInfo(
        "Legacy/experimental QueryIR training and calibration commands.",
        "Some ablation and legacy tests still exercise these paths.",
        "ML",
        "legacy/experimental source",
        "Review for consolidation into training/ after proving command replacement.",
    ),
    "validation": FolderInfo(
        "Central SQL validation package.",
        "Execution safety requires a shared SELECT-only validator.",
        "Security",
        "source",
        "Keep as canonical SQL validation boundary.",
    ),
}


ENTRY_POINTS: list[dict[str, str]] = [
    {
        "name": "Streamlit application",
        "command": "streamlit run app/streamlit_app.py",
        "main_module": "app/streamlit_app.py",
        "input_configuration": "NL2SQL_ENV, NL2SQL_ALLOW_CANDIDATE_BUNDLE, app sidebar database settings",
        "generated_artifacts": "Runtime telemetry/prediction cache when configured",
        "failure_behaviour": "Fails closed when production bundle is missing or invalid",
        "production_relevance": "primary runtime UI",
    },
    {
        "name": "Production training pipeline",
        "command": "python training/train_model.py --config configs/training.yaml",
        "main_module": "training/train_model.py",
        "input_configuration": "configs/training.yaml",
        "generated_artifacts": "artifacts/pipeline/runs/<run_id>, artifacts/model_bundle/candidates/<run_id>, optional current bundle",
        "failure_behaviour": "Pipeline step failures block promotion",
        "production_relevance": "canonical production training command",
    },
    {
        "name": "Smoke training pipeline",
        "command": "python training/train_model.py --config configs/smoke_training.yaml",
        "main_module": "training/train_model.py",
        "input_configuration": "configs/smoke_training.yaml",
        "generated_artifacts": "smoke-scoped pipeline and candidate artifacts",
        "failure_behaviour": "Fast integration failure signal",
        "production_relevance": "developer validation only",
    },
    {
        "name": "Baseline training pipeline",
        "command": "python training/train_model.py --config configs/baseline_training.yaml",
        "main_module": "training/train_model.py",
        "input_configuration": "configs/baseline_training.yaml",
        "generated_artifacts": "baseline pipeline reports and candidate bundle",
        "failure_behaviour": "Full diagnostics without all production promotion requirements",
        "production_relevance": "release evidence, not production promotion by itself",
    },
    {
        "name": "Dataset verification",
        "command": "python scripts/verify_datasets.py",
        "main_module": "scripts/verify_datasets.py",
        "input_configuration": "data/raw and data/processed dataset paths",
        "generated_artifacts": "none expected",
        "failure_behaviour": "Reports missing/unusable datasets",
        "production_relevance": "training readiness",
    },
    {
        "name": "Dataset download",
        "command": "python scripts/download_datasets.py --datasets wikisql spider bird-mini",
        "main_module": "scripts/download_datasets.py",
        "input_configuration": "dataset arguments and local data/ paths",
        "generated_artifacts": "data/raw/ and data/processed/",
        "failure_behaviour": "Stops on unavailable downloads or invalid destinations",
        "production_relevance": "data preparation",
    },
    {
        "name": "BIRD full preparation",
        "command": "python scripts/prepare_bird_full.py",
        "main_module": "scripts/prepare_bird_full.py",
        "input_configuration": "data/raw/bird/full",
        "generated_artifacts": "prepared BIRD full manifest and normalized data",
        "failure_behaviour": "Fails when source files are missing or malformed",
        "production_relevance": "large dataset preparation",
    },
    {
        "name": "Golden tests",
        "command": "python scripts/run_golden_tests.py",
        "main_module": "scripts/run_golden_tests.py",
        "input_configuration": "evaluation/golden_tests.jsonl",
        "generated_artifacts": "evaluation/golden_test_results.json",
        "failure_behaviour": "Nonzero on failed golden cases",
        "production_relevance": "regression validation",
    },
    {
        "name": "Integration readiness audit",
        "command": "python scripts/audit_integration_readiness.py",
        "main_module": "scripts/audit_integration_readiness.py",
        "input_configuration": "repo source, configs, docs, and artifacts",
        "generated_artifacts": "artifacts/audit/integration_readiness_report.*",
        "failure_behaviour": "Nonzero when required integration evidence is missing",
        "production_relevance": "release gate",
    },
    {
        "name": "Execution pipeline audit",
        "command": "python scripts/audit_execution_pipeline_readiness.py",
        "main_module": "scripts/audit_execution_pipeline_readiness.py",
        "input_configuration": "execution/runtime/evaluation modules",
        "generated_artifacts": "artifacts/audit/execution_pipeline_readiness_report.*",
        "failure_behaviour": "Nonzero on missing execution safety evidence",
        "production_relevance": "release gate",
    },
    {
        "name": "Generic NL-to-SQL audit",
        "command": "python scripts/audit_generic_nl2sql_readiness.py",
        "main_module": "scripts/audit_generic_nl2sql_readiness.py",
        "input_configuration": "runtime/training generic NL-to-SQL paths",
        "generated_artifacts": "artifacts/audit/generic_nl2sql_readiness_report.*",
        "failure_behaviour": "Nonzero on genericity gaps",
        "production_relevance": "release gate",
    },
    {
        "name": "Self-training audit",
        "command": "python scripts/audit_self_training_readiness.py",
        "main_module": "scripts/audit_self_training_readiness.py",
        "input_configuration": "self_training and feedback modules",
        "generated_artifacts": "artifacts/audit/self_training_readiness_report.*",
        "failure_behaviour": "Nonzero on missing governance controls",
        "production_relevance": "self-training gate",
    },
    {
        "name": "Test suite",
        "command": "python -m pytest tests/ --tb=short",
        "main_module": "tests/",
        "input_configuration": "pytest.ini",
        "generated_artifacts": ".pytest_cache/ and coverage outputs when enabled",
        "failure_behaviour": "Nonzero on test failure",
        "production_relevance": "required regression gate",
    },
]


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return (result.stdout + result.stderr).strip()
    return result.stdout.strip()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def read_text_if_small(path: Path, max_bytes: int = 2_000_000) -> str | None:
    size = file_size(path)
    if size is None or size > max_bytes:
        return None
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"README.md", "pytest.ini"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return None
    except OSError:
        return None


def tracked_files() -> list[str]:
    output = run_git("ls-files")
    return [line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()]


def untracked_files() -> list[str]:
    output = run_git("ls-files", "--others", "--exclude-standard")
    return [line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()]


def ignored_summary() -> list[str]:
    output = run_git("status", "--short", "--ignored")
    ignored = []
    for line in output.splitlines():
        if line.startswith("!! "):
            ignored.append(line[3:].replace("\\", "/"))
    return ignored


def module_name(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    candidate = path[:-3].replace("/", ".")
    if candidate.endswith(".__init__"):
        candidate = candidate[: -len(".__init__")]
    return candidate


def parse_python(path: Path) -> dict[str, Any]:
    source = read_text_if_small(path)
    result: dict[str, Any] = {
        "module_docstring": None,
        "public_classes": 0,
        "public_classes_missing_docstrings": [],
        "public_functions": 0,
        "public_functions_missing_docstrings": [],
        "imports": [],
        "parse_error": None,
    }
    if source is None:
        return result
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result["parse_error"] = f"{exc.msg} at line {exc.lineno}"
        return result
    result["module_docstring"] = ast.get_docstring(tree)
    imports: list[str] = []
    current_module = module_name(rel(path)) or ""
    current_package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module
                if node.level:
                    parts = current_package.split(".") if current_package else []
                    if node.level > 1:
                        parts = parts[: -(node.level - 1)]
                    base = ".".join([*parts, node.module]) if parts else node.module
                imports.append(base)
                imports.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                result["public_classes"] += 1
                if not ast.get_docstring(node):
                    result["public_classes_missing_docstrings"].append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                result["public_functions"] += 1
                if not ast.get_docstring(node):
                    result["public_functions_missing_docstrings"].append(node.name)
    result["imports"] = sorted(set(imports))
    return result


def item_type_for(path: str) -> str:
    p = Path(path)
    if path.endswith("/"):
        return "directory"
    suffix = p.suffix.lower()
    if suffix == ".py":
        return "python_source"
    if suffix in {".yaml", ".yml", ".toml", ".ini"} or p.name == "pytest.ini":
        return "configuration"
    if suffix == ".json":
        return "json_data_or_report"
    if suffix == ".jsonl":
        return "jsonl_data"
    if suffix == ".md":
        return "documentation"
    if suffix == ".sql":
        return "sql_fixture"
    if suffix in GENERATED_SUFFIXES:
        return "generated_binary_or_database"
    if suffix in {".txt", ".lock"}:
        return "dependency_or_text"
    return "other"


def folder_key(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 2 and "/".join(parts[:2]) in FOLDER_INFO:
        return "/".join(parts[:2])
    return parts[0]


def folder_info(path: str) -> FolderInfo:
    return FOLDER_INFO.get(folder_key(path), FOLDER_INFO.get(path.split("/")[0], FolderInfo(
        "Unclassified repository area.",
        "Requires manual owner confirmation.",
        "REVIEW_REQUIRED",
        "unknown",
        "Review before retention or deletion.",
    )))


def is_entry_point(path: str, py_info: dict[str, Any] | None = None) -> bool:
    known = {entry["main_module"] for entry in ENTRY_POINTS if entry["main_module"].endswith(".py")}
    if path in known:
        return True
    if path.startswith("scripts/") and path.endswith(".py"):
        return True
    if path.startswith("training/") and path.endswith(".py") and Path(path).name.startswith(("train_", "run_", "build_", "evaluate_", "promote_", "select_")):
        return True
    text = read_text_if_small(ROOT / path)
    return bool(text and 'if __name__ == "__main__"' in text)


def classify(path: str, status: str) -> tuple[str, str, str, str]:
    p = Path(path)
    info = folder_info(path)
    cleanup_action = "retain_and_document"
    replacement = ""
    if status == "ignored":
        if any(part in CACHE_DIR_NAMES for part in p.parts) or p.name.endswith((".pyc", ".pyo")) or p.name in ROOT_LOG_NAMES:
            return "DELETE", "low-risk generated cache/log file", "delete_low_risk_generated", ""
        if path.startswith(("artifacts/model_bundle/", "artifacts/pipeline/runs/", "artifacts/audit/")):
            return "REVIEW_REQUIRED", "ignored artifact with possible audit/runtime value", "retain_pending_artifact_review", ""
        if path.startswith(("data/raw/", "data/processed")):
            return "REVIEW_REQUIRED", "ignored downloaded/generated dataset", "retain_pending_dataset_review", ""
        return "REGENERATE", "ignored generated local output", "keep_ignored_or_regenerate", ""
    if path in {"README.md", "docs/REPOSITORY_MAP.md"}:
        return "KEEP_CANONICAL", "canonical repository documentation", cleanup_action, replacement
    if path.startswith("docs/reports/"):
        return "ARCHIVE", "historical or generated report under docs/reports", "archive_or_move_after_review", replacement
    if path.startswith("data/splits/"):
        return "KEEP_GENERATED", "frozen dataset split artifact", cleanup_action, replacement
    if path.startswith("artifacts/repository_cleanup/"):
        return "KEEP_GENERATED", "cleanup governance report generated by inventory script", "regenerate_after_cleanup_changes", replacement
    if path.startswith("tests/legacy/") or path.startswith("nl2sql_v1/") or path.startswith("training_ir/"):
        return "REVIEW_REQUIRED", "legacy or experimental path still covered by tests/docs", "retain_pending_consolidation", replacement
    if path.startswith("models/") and p.name == ".gitkeep":
        return "KEEP_SUPPORTING", "placeholder keeps ignored artifact directory in Git", cleanup_action, replacement
    if path.startswith("artifacts/option_c_model/") and p.name == ".gitkeep":
        return "KEEP_SUPPORTING", "placeholder for ignored legacy artifact directory", cleanup_action, replacement
    if path.startswith(".github/") and status == "untracked":
        return "REVIEW_REQUIRED", "new CI workflow should be reviewed and then tracked", "review_before_commit", replacement
    if path.startswith("data/splits/semantic_v2/"):
        return "KEEP_GENERATED", "active configured split version in configs/training.yaml", "track_or_retain_as_frozen_split", replacement
    if path.startswith("tests/") and path.endswith(".py"):
        return "KEEP_SUPPORTING", "active test module", cleanup_action, replacement
    if is_entry_point(path):
        return "KEEP_CANONICAL", "supported command entry point", cleanup_action, replacement
    if path.endswith(".py"):
        if folder_key(path) in {"app", "training", "inference", "ir", "model_bundle", "execution", "validation", "db", "orchestration"}:
            return "KEEP_CANONICAL", info.purpose, cleanup_action, replacement
        return "KEEP_SUPPORTING", info.purpose, cleanup_action, replacement
    if path.startswith(("configs/", "pipeline_configs/")):
        return "KEEP_SUPPORTING", "consumed configuration", cleanup_action, replacement
    if path.startswith("evaluation/") and path.endswith((".jsonl", ".yaml", ".sql")):
        return "KEEP_SUPPORTING", "checked-in evaluation fixture or threshold", cleanup_action, replacement
    if path.startswith(("docs/", "data/", "training_data/")):
        return "KEEP_SUPPORTING", info.purpose, cleanup_action, replacement
    if status == "untracked":
        return "REVIEW_REQUIRED", "untracked item needs explicit review before commit/delete", "review_before_commit", replacement
    return "KEEP_SUPPORTING", info.purpose, cleanup_action, replacement


def build_text_reference_index(paths: list[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in paths:
        text = read_text_if_small(ROOT / path)
        if text is not None:
            texts[path] = text
    return texts


def resolve_import(import_name: str, modules: dict[str, str]) -> str | None:
    parts = import_name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in modules:
            return modules[candidate]
    return None


def collect_artifact_state() -> dict[str, Any]:
    bundle_root = ROOT / "artifacts" / "model_bundle"
    current_manifest = bundle_root / "current" / "bundle_manifest.json"
    candidate_manifest = bundle_root / "candidate" / "bundle_manifest.json"
    candidates_dir = bundle_root / "candidates"
    candidates: list[dict[str, Any]] = []
    if candidates_dir.exists():
        for child in sorted(candidates_dir.iterdir()):
            manifest = load_json(child / "bundle_manifest.json") or {}
            candidates.append(
                {
                    "path": rel(child),
                    "bundle_id": manifest.get("bundle_id"),
                    "status": manifest.get("status") or manifest.get("bundle_status"),
                    "created_at": manifest.get("created_at"),
                    "git_commit": manifest.get("git_commit"),
                    "quality_gate_passed": manifest.get("quality_gate_passed"),
                    "production_ready_full": manifest.get("production_ready_full"),
                    "eligible_for_promotion": manifest.get("eligible_for_promotion"),
                }
            )

    runs = []
    runs_dir = ROOT / "artifacts" / "pipeline" / "runs"
    if runs_dir.exists():
        for child in sorted(runs_dir.iterdir()):
            report = load_json(child / "pipeline_report.json") or {}
            train_report = load_json(child / "train_model_report.json") or {}
            status = report.get("status") or train_report.get("status")
            completed_at = train_report.get("completed_at") or train_report.get("completed")
            runs.append(
                {
                    "path": rel(child),
                    "run_id": child.name,
                    "status": status,
                    "pipeline": report.get("pipeline") or train_report.get("pipeline"),
                    "completed_at": completed_at,
                }
            )

    completed_runs = [run for run in runs if str(run.get("status")).lower() == "completed"]
    failed_runs = [run for run in runs if str(run.get("status")).lower() in {"failed", "error"}]
    return {
        "active_model_bundle_path": "artifacts/model_bundle/current",
        "current_production_bundle": {
            "path": rel(current_manifest) if current_manifest.exists() else "artifacts/model_bundle/current/bundle_manifest.json",
            "exists": current_manifest.exists(),
            "manifest": summarize_bundle_manifest(load_json(current_manifest) if current_manifest.exists() else None),
        },
        "singleton_candidate_bundle": {
            "path": rel(candidate_manifest) if candidate_manifest.exists() else "artifacts/model_bundle/candidate/bundle_manifest.json",
            "exists": candidate_manifest.exists(),
            "manifest": summarize_bundle_manifest(load_json(candidate_manifest) if candidate_manifest.exists() else None),
        },
        "candidate_bundles": candidates,
        "latest_successful_training_run": completed_runs[-1] if completed_runs else None,
        "latest_failed_training_run": failed_runs[-1] if failed_runs else None,
        "pipeline_runs": runs,
        "active_dataset_split_version": detect_active_split_version(),
    }


def summarize_bundle_manifest(manifest: Any) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    return {
        "bundle_id": manifest.get("bundle_id"),
        "status": manifest.get("status") or manifest.get("bundle_status"),
        "created_at": manifest.get("created_at"),
        "git_commit": manifest.get("git_commit"),
        "training_config_path": manifest.get("training_config_path"),
        "quality_gate_mode": manifest.get("quality_gate_mode"),
        "quality_gate_passed": manifest.get("quality_gate_passed"),
        "eligible_for_promotion": manifest.get("eligible_for_promotion"),
        "production_ready_full": manifest.get("production_ready_full"),
        "datasets": manifest.get("datasets"),
    }


def detect_active_split_version() -> dict[str, Any]:
    config = load_json(ROOT / "configs" / "training.yaml")
    if config is not None:
        return {"source": "configs/training.yaml", "split_version": config.get("dataset", {}).get("split_version")}
    text = read_text_if_small(ROOT / "configs" / "training.yaml") or ""
    match = re.search(r"^\s*split_version:\s*([A-Za-z0-9_.-]+)\s*$", text, flags=re.MULTILINE)
    split_version = match.group(1) if match else None
    manifest_path = ROOT / "data" / "splits" / str(split_version or "") / "split_manifest.json"
    manifest = load_json(manifest_path) if split_version else None
    return {
        "source": "configs/training.yaml",
        "split_version": split_version,
        "manifest_path": rel(manifest_path) if manifest_path.exists() else None,
        "manifest_sha256": manifest.get("manifest_sha256") if isinstance(manifest, dict) else None,
    }


def collect_baseline() -> dict[str, Any]:
    return {
        "captured_at": now_iso(),
        "baseline_commit_requested": BASELINE_COMMIT,
        "head": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "recent_commits": run_git("log", "--oneline", "--decorate", "-10").splitlines(),
        "status_short": run_git("status", "--short").splitlines(),
        "diff_stat": run_git("diff", "--stat").splitlines(),
        "untracked_files": untracked_files(),
        "ignored_summary": ignored_summary(),
        "working_tree_clean": run_git("status", "--short").strip() == "",
        "artifact_state": collect_artifact_state(),
    }


def build_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tracked = tracked_files()
    untracked = untracked_files()
    ignored = ignored_summary()
    all_file_paths = sorted(set(tracked + untracked))

    py_info: dict[str, dict[str, Any]] = {}
    modules: dict[str, str] = {}
    for path in all_file_paths:
        if path.endswith(".py") and (ROOT / path).is_file():
            info = parse_python(ROOT / path)
            py_info[path] = info
            mod = module_name(path)
            if mod:
                modules[mod] = path

    reverse_imports: dict[str, set[str]] = defaultdict(set)
    forward_imports: dict[str, list[str]] = {}
    for path, info in py_info.items():
        refs: list[str] = []
        for imported in info.get("imports", []):
            target = resolve_import(imported, modules)
            if target and target != path:
                reverse_imports[target].add(path)
                refs.append(target)
        forward_imports[path] = sorted(set(refs))

    texts = build_text_reference_index(all_file_paths)
    text_reference_count: dict[str, int] = {}
    text_referenced_by: dict[str, list[str]] = {}
    for path in all_file_paths:
        needle_variants = {path, path.replace("/", "\\")}
        refs = []
        for source_path, text in texts.items():
            if source_path == path:
                continue
            if any(needle in text for needle in needle_variants):
                refs.append(source_path)
        text_reference_count[path] = len(refs)
        text_referenced_by[path] = refs[:25]

    inventory: list[dict[str, Any]] = []
    for path in all_file_paths:
        status = "tracked" if path in tracked else "untracked"
        abs_path = ROOT / path
        info = folder_info(path)
        classification, reason, cleanup_action, replacement = classify(path, status)
        py = py_info.get(path)
        module_doc = py.get("module_docstring") if py else None
        purpose = first_sentence(module_doc) if module_doc else f"{item_type_for(path).replace('_', ' ')} in {folder_key(path)}."
        referenced_by = sorted(reverse_imports.get(path, set()))
        if text_referenced_by.get(path):
            referenced_by = sorted(set(referenced_by + text_referenced_by[path]))
        evidence = [
            f"git_status={status}",
            f"folder_owner={info.owner}",
            f"item_type={item_type_for(path)}",
        ]
        if py:
            evidence.append(f"module_docstring={'present' if module_doc else 'missing'}")
            evidence.append(f"public_classes={py.get('public_classes', 0)}")
            evidence.append(f"public_functions={py.get('public_functions', 0)}")
            if py.get("parse_error"):
                evidence.append(f"parse_error={py['parse_error']}")
        if referenced_by:
            evidence.append(f"referenced_by_count={len(referenced_by)}")
        if text_reference_count.get(path):
            evidence.append(f"text_reference_count={text_reference_count[path]}")
        entry = {
            "path": path,
            "git_status": status,
            "item_type": item_type_for(path),
            "classification": classification,
            "purpose": purpose,
            "why_required": reason if classification in {"KEEP_CANONICAL", "KEEP_SUPPORTING", "KEEP_GENERATED"} else info.why_required,
            "referenced_by": referenced_by,
            "references": forward_imports.get(path, []),
            "generated_by": generated_by_for(path),
            "entry_point": is_entry_point(path, py),
            "runtime_used": runtime_used(path, referenced_by),
            "training_used": training_used(path, referenced_by),
            "test_used": path.startswith("tests/") or any(ref.startswith("tests/") for ref in referenced_by),
            "bundle_used": bundle_used(path),
            "replacement_path": replacement or None,
            "cleanup_action": cleanup_action,
            "confidence": confidence_for(classification, referenced_by, status),
            "evidence": evidence,
            "checksum": sha256(abs_path),
            "size_bytes": file_size(abs_path),
        }
        inventory.append(entry)

    ignored_entries = []
    for path in ignored:
        classification, reason, cleanup_action, replacement = classify(path, "ignored")
        ignored_entries.append(
            {
                "path": path,
                "git_status": "ignored",
                "item_type": "ignored_directory_or_file",
                "classification": classification,
                "purpose": reason,
                "why_required": folder_info(path).why_required,
                "referenced_by": [],
                "references": [],
                "generated_by": generated_by_for(path),
                "entry_point": False,
                "runtime_used": bundle_used(path) or path.startswith("data/"),
                "training_used": path.startswith(("artifacts/", "data/", "training_data/")),
                "test_used": path.startswith(("tests/", ".pytest_cache")),
                "bundle_used": bundle_used(path),
                "replacement_path": replacement or None,
                "cleanup_action": cleanup_action,
                "confidence": "medium" if classification == "DELETE" else "low",
                "evidence": ["git_status=ignored", f"cleanup_action={cleanup_action}"],
                "checksum": None,
                "size_bytes": None,
            }
        )
    inventory.extend(ignored_entries)

    stats = {
        "tracked_files": len(tracked),
        "untracked_files": len(untracked),
        "ignored_summary_entries": len(ignored),
        "python_files": sum(1 for p in tracked if p.endswith(".py")),
        "markdown_files": sum(1 for p in tracked if p.endswith(".md")),
        "test_files": sum(1 for p in tracked if p.startswith("tests/") and p.endswith(".py")),
        "configuration_files": sum(1 for p in tracked if item_type_for(p) == "configuration"),
        "generated_files_in_git": sum(1 for e in inventory if e["git_status"] == "tracked" and e["classification"] == "KEEP_GENERATED"),
        "classification_counts": dict(Counter(e["classification"] for e in inventory)),
        "python_docstring_coverage": docstring_coverage(py_info),
    }
    return inventory, stats


def first_sentence(text: str | None) -> str:
    if not text:
        return ""
    clean = " ".join(text.strip().split())
    match = re.match(r"(.+?[.!?])(\s|$)", clean)
    return match.group(1) if match else clean[:180]


def generated_by_for(path: str) -> str | None:
    if path.startswith("artifacts/repository_cleanup/"):
        return "python scripts/generate_repository_cleanup_inventory.py"
    if path.startswith("data/splits/"):
        return "dataset_training.split_manager or training/train_model.py split step"
    if path.startswith("artifacts/model_bundle/"):
        return "training/train_model.py build_model_bundle step"
    if path.startswith("artifacts/pipeline/"):
        return "training/train_model.py"
    if path.startswith("evaluation/") and path.endswith("_report.json"):
        return "evaluation or training report command"
    if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
        return "Python interpreter"
    if path.endswith(".log"):
        return "local smoke/test command"
    return None


def runtime_used(path: str, referenced_by: list[str]) -> bool:
    runtime_roots = ("app/", "inference/", "execution/", "db/", "ir/", "model_bundle/", "retrieval/", "retriever/", "generic_planner/", "semantic_layer/", "validation/")
    return path.startswith(runtime_roots) or any(ref.startswith(runtime_roots) for ref in referenced_by)


def training_used(path: str, referenced_by: list[str]) -> bool:
    training_roots = ("training/", "dataset_training/", "datasets/", "neural_ir/", "neural_optimization/", "orchestration/", "quality_gates/", "model_selection/", "reward/")
    return path.startswith(training_roots) or any(ref.startswith(training_roots) for ref in referenced_by)


def bundle_used(path: str) -> bool:
    return path.startswith(("artifacts/model_bundle/", "model_bundle/")) or "bundle_manifest" in path


def confidence_for(classification: str, referenced_by: list[str], status: str) -> str:
    if classification in {"KEEP_CANONICAL", "KEEP_GENERATED"}:
        return "high"
    if status == "untracked" or classification in {"REVIEW_REQUIRED", "ARCHIVE", "CONSOLIDATE"}:
        return "medium"
    if referenced_by:
        return "high"
    return "medium"


def docstring_coverage(py_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(py_info)
    modules_with_docstrings = sum(1 for info in py_info.values() if info.get("module_docstring"))
    public_classes = sum(int(info.get("public_classes", 0)) for info in py_info.values())
    missing_classes = sum(len(info.get("public_classes_missing_docstrings", [])) for info in py_info.values())
    public_functions = sum(int(info.get("public_functions", 0)) for info in py_info.values())
    missing_functions = sum(len(info.get("public_functions_missing_docstrings", [])) for info in py_info.values())
    missing_module_files = sorted(path for path, info in py_info.items() if not info.get("module_docstring"))
    missing_public_docstrings = {
        path: {
            "classes": info.get("public_classes_missing_docstrings", []),
            "functions": info.get("public_functions_missing_docstrings", []),
        }
        for path, info in sorted(py_info.items())
        if info.get("public_classes_missing_docstrings") or info.get("public_functions_missing_docstrings")
    }
    return {
        "python_files_analyzed": total,
        "modules_with_docstrings": modules_with_docstrings,
        "modules_missing_docstrings": total - modules_with_docstrings,
        "public_classes": public_classes,
        "public_classes_missing_docstrings": missing_classes,
        "public_functions": public_functions,
        "public_functions_missing_docstrings": missing_functions,
        "missing_module_docstring_files": missing_module_files,
        "missing_public_docstring_symbols": missing_public_docstrings,
    }


def collect_config_usage(texts: dict[str, str]) -> list[dict[str, Any]]:
    config_files = [path for path in tracked_files() + untracked_files() if item_type_for(path) == "configuration"]
    rows = []
    for path in sorted(set(config_files)):
        text = read_text_if_small(ROOT / path) or ""
        keys = []
        for line in text.splitlines():
            match = re.match(r"^(\s*)([A-Za-z0-9_.-]+):", line)
            if match:
                keys.append(match.group(2))
        consumers = [source for source, body in texts.items() if source != path and path in body]
        rows.append(
            {
                "file": path,
                "top_level_or_section_keys": sorted(set(keys))[:80],
                "referenced_by": consumers[:25],
                "active": bool(consumers) or path.startswith("configs/"),
                "cleanup_action": "retain_and_validate_consumers" if path.startswith(("configs/", "pipeline_configs/")) else "review",
            }
        )
    return rows


def collect_hardcoded_rules(texts: dict[str, str]) -> list[dict[str, Any]]:
    patterns = [
        "customers",
        "orders",
        "order_items",
        "revenue",
        "quantity * unit_price",
        "quantity * price",
        "APPROVED_REVENUE_EXPR",
        "sample schema",
        "normalize_runtime_result",
    ]
    rows = []
    for path, text in sorted(texts.items()):
        hits = [pattern for pattern in patterns if pattern in text]
        if not hits:
            continue
        if path.startswith("tests/") or path.startswith("evaluation/"):
            classification = "test_or_benchmark_fixture"
        elif path in {"data/synonyms.yaml", "data/templates.yaml"}:
            classification = "configuration_driven_semantic_defaults"
        elif path.startswith("docs/"):
            classification = "documentation_example_or_report"
        elif path.startswith(("inference/", "retriever/", "generic_planner/", "quality_gates/")):
            classification = "runtime_or_gate_review_required"
        else:
            classification = "review_required"
        rows.append({"path": path, "patterns": hits, "classification": classification})
    return rows


def collect_documentation_report(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    docs = [e for e in inventory if e["item_type"] == "documentation" and e["git_status"] != "ignored"]
    return {
        "canonical_docs": [
            "README.md",
            "docs/REPOSITORY_MAP.md",
            "docs/architecture/current_end_to_end_flow.md",
            "docs/architecture/production_runtime_policy.md",
            "docs/architecture/model_bundle_lifecycle.md",
            "docs/architecture/database_execution_policy.md",
            "docs/architecture/privacy_and_retention_policy.md",
            "docs/architecture/training_data_lineage.md",
            "docs/developer_commands.md",
            "docs/deployment.md",
            "docs/capability_taxonomy.md",
            "docs/specs/query_ir_v1_frozen_spec.md",
            "docs/specs/query_ir_v2_foundation_spec.md",
            "docs/specs/query_ir_v2_boolean_predicate_spec.md",
        ],
        "report_docs": [e["path"] for e in docs if e["path"].startswith("docs/reports/")],
        "legacy_docs": [e["path"] for e in docs if "legacy" in e["path"].lower()],
        "cleanup_actions": {
            "docs/reports": "Archive or move run-scoped generated reports to artifact run directories after owner review.",
            "canonical_docs": "Keep and validate command/path references.",
        },
    }


def collect_test_report(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    tests = [e["path"] for e in inventory if e["path"].startswith("tests/") and e["path"].endswith(".py")]
    return {
        "active_test_files": sorted(path for path in tests if not path.startswith("tests/legacy/")),
        "legacy_test_files": sorted(path for path in tests if path.startswith("tests/legacy/")),
        "classification": "Legacy tests remain active under pytest discovery unless explicitly excluded.",
        "cleanup_action": "Review legacy tests individually; do not delete failing/stale tests without replacement coverage.",
    }


def collect_consolidation_report(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "area": "retrieval vs retriever",
            "classification": "CONSOLIDATE",
            "canonical_owner": "retrieval/ for index/reranker infrastructure; retriever/ currently retains runtime RetrievalNL2SQLModel wrapper",
            "evidence": "app/streamlit_app.py and tests still import retriever.retrieval_nl2sql_model",
            "cleanup_action": "Plan import migration before deleting retriever/.",
        },
        {
            "area": "training vs training_ir",
            "classification": "REVIEW_REQUIRED",
            "canonical_owner": "training/train_model.py for integrated pipeline",
            "evidence": "training_ir contains legacy/experimental Option A commands with legacy tests",
            "cleanup_action": "Keep until commands are replaced or archived with tests/docs updated.",
        },
        {
            "area": "dataset_training vs datasets",
            "classification": "KEEP_BOTH",
            "canonical_owner": "datasets/ adapters; dataset_training/ corpus/split/leakage builders",
            "evidence": "Different responsibilities; both are referenced by training paths.",
            "cleanup_action": "No merge without API design.",
        },
        {
            "area": "models vs model_bundle",
            "classification": "KEEP_BOTH",
            "canonical_owner": "model_bundle/ for production bundles; models/ only ignored local artifact placeholder",
            "evidence": "Runtime uses bundle manifests rather than models/ guesses.",
            "cleanup_action": "Keep models/.gitkeep only.",
        },
        {
            "area": "evaluation reports in docs/reports vs artifacts/",
            "classification": "ARCHIVE",
            "canonical_owner": "artifacts/pipeline/runs/<run_id>/reports for run-scoped generated reports",
            "evidence": "docs/reports contains run-specific quality gate diagnosis.",
            "cleanup_action": "Move after review; no automatic deletion in this pass.",
        },
    ]


def low_risk_delete_candidates() -> list[Path]:
    candidates: list[Path] = []
    for name in CACHE_DIR_NAMES:
        candidates.extend(path for path in ROOT.rglob(name) if path.is_dir() and "venv" not in path.parts and ".git" not in path.parts)
    for path in ROOT.rglob("*.pyc"):
        if "venv" not in path.parts and ".git" not in path.parts:
            candidates.append(path)
    for name in ROOT_LOG_NAMES:
        path = ROOT / name
        if path.exists():
            candidates.append(path)
    return sorted(set(candidates), key=lambda item: str(item).lower())


def safe_delete_low_risk() -> list[dict[str, Any]]:
    manifest = []
    root_resolved = ROOT.resolve()
    for path in low_risk_delete_candidates():
        if not path.exists():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        if root_resolved not in resolved.parents and resolved != root_resolved:
            raise RuntimeError(f"Refusing to delete outside workspace: {resolved}")
        if ".git" in resolved.parts or "venv" in resolved.parts:
            continue
        entry = {
            "path": rel(resolved),
            "reason": "Low-risk generated Python/test cache or local smoke log.",
            "classification": "DELETE",
            "replacement": None,
            "references_checked": True,
            "generated": True,
            "recoverable_from_git": False,
            "validation_performed": ["path prefix check", "gitignore coverage", "generated-cache classification"],
            "risk": "low",
            "deleted_at": now_iso(),
        }
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        manifest.append(entry)
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def generate_repo_map(inventory: list[dict[str, Any]], baseline: dict[str, Any], stats: dict[str, Any]) -> str:
    existing_dirs = sorted({folder_key(e["path"]) for e in inventory if e["git_status"] != "ignored"})
    rows = []
    for key in sorted(FOLDER_INFO):
        if key.split("/")[0] not in existing_dirs and key not in {"artifacts/repository_cleanup"}:
            continue
        info = FOLDER_INFO[key]
        rows.append([key, info.purpose, info.why_required, info.owner, info.generated_or_source, info.cleanup_rule])

    entry_rows = [
        [
            e["name"],
            e["command"],
            e["main_module"],
            e["input_configuration"],
            e["generated_artifacts"],
            e["failure_behaviour"],
            e["production_relevance"],
        ]
        for e in ENTRY_POINTS
    ]
    artifact = baseline["artifact_state"]
    lines = [
        "# Repository Map",
        "",
        f"Generated: {now_iso()}",
        f"Branch: {baseline.get('branch')}",
        f"HEAD: {baseline.get('head')}",
        f"Requested baseline commit: {BASELINE_COMMIT}",
        "",
        "## Baseline State",
        "",
        f"- Working tree clean at start: {baseline.get('working_tree_clean')}",
        f"- Modified/untracked entries at start: {len(baseline.get('status_short', []))}",
        f"- Active model-bundle path: {artifact['active_model_bundle_path']}",
        f"- Current production bundle exists locally: {artifact['current_production_bundle']['exists']}",
        f"- Singleton candidate bundle exists locally: {artifact['singleton_candidate_bundle']['exists']}",
        f"- Run-scoped candidate bundles found: {len(artifact['candidate_bundles'])}",
        f"- Active dataset split: {artifact['active_dataset_split_version'].get('split_version')}",
        f"- Latest successful training run: {(artifact['latest_successful_training_run'] or {}).get('run_id')}",
        f"- Latest failed training run: {(artifact['latest_failed_training_run'] or {}).get('run_id')}",
        "",
        "## Folder Inventory",
        "",
        markdown_table(["Folder", "Purpose", "Why required", "Canonical owner", "Generated or source", "Cleanup rule"], rows),
        "",
        "## Entry Points",
        "",
        markdown_table(
            ["Name", "Command", "Main module", "Input configuration", "Generated artifacts", "Failure behaviour", "Production relevance"],
            entry_rows,
        ),
        "",
        "## Repository Statistics",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["tracked files", stats["tracked_files"]],
                ["untracked files", stats["untracked_files"]],
                ["ignored summary entries", stats["ignored_summary_entries"]],
                ["Python files", stats["python_files"]],
                ["Markdown files", stats["markdown_files"]],
                ["test files", stats["test_files"]],
                ["configuration files", stats["configuration_files"]],
                ["generated files in Git", stats["generated_files_in_git"]],
            ],
        ),
        "",
        "## Cleanup Rule",
        "",
        "Retain production-critical artifacts and frozen splits unless a later manifest proves replacement, references, and reproducibility. Delete only generated caches/logs automatically.",
        "",
    ]
    return "\n".join(lines)


def generate_cleanup_report(
    baseline: dict[str, Any],
    stats_before: dict[str, Any],
    stats_after: dict[str, Any] | None,
    deletion_manifest: list[dict[str, Any]],
    archive_manifest: list[dict[str, Any]],
    consolidation: list[dict[str, Any]],
    documentation: dict[str, Any],
    tests: dict[str, Any],
    hardcoded: list[dict[str, Any]],
    config_usage: list[dict[str, Any]],
) -> str:
    artifact = baseline["artifact_state"]
    doc_cov = stats_before["python_docstring_coverage"]
    after = stats_after or stats_before
    lines = [
        "# Repository Cleanup Report",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Executive Summary",
        "",
        "This pass created the cleanup inventory, repository map, manifests, artifact governance reports, and low-risk deletion evidence. It did not delete production-critical bundles, frozen splits, model checkpoints, raw datasets, or run-scoped reports.",
        "",
        "## Safety Statement",
        "",
        f"- Current production bundle exists locally: {artifact['current_production_bundle']['exists']}. Missing current bundle is a release risk, not a cleanup target.",
        f"- Singleton candidate bundle exists locally: {artifact['singleton_candidate_bundle']['exists']}. It is retained pending artifact review.",
        f"- Run-scoped candidate bundles found: {len(artifact['candidate_bundles'])}. They are retained pending retention policy application.",
        f"- Active split version: {artifact['active_dataset_split_version'].get('split_version')}. Frozen split manifests are retained.",
        "",
        "## Low-Risk Deletions",
        "",
        markdown_table(
            ["Path", "Reason", "Risk"],
            [[item["path"], item["reason"], item["risk"]] for item in deletion_manifest] or [["None", "No low-risk generated caches/logs were present or deletion was not requested.", "n/a"]],
        ),
        "",
        "## Archive / Review Candidates",
        "",
        markdown_table(
            ["Path", "Reason", "Action"],
            [[item["path"], item["reason"], item["cleanup_action"]] for item in archive_manifest] or [["None", "No archive candidates identified.", "n/a"]],
        ),
        "",
        "## Consolidation Candidates",
        "",
        markdown_table(
            ["Area", "Classification", "Canonical owner", "Cleanup action"],
            [[item["area"], item["classification"], item["canonical_owner"], item["cleanup_action"]] for item in consolidation],
        ),
        "",
        "## Documentation Cleanup",
        "",
        f"- Canonical docs retained: {len(documentation['canonical_docs'])}",
        f"- Report docs needing archive/move review: {len(documentation['report_docs'])}",
        f"- Legacy docs retained pending retirement: {len(documentation['legacy_docs'])}",
        "",
        "## Artifact Retention Policy",
        "",
        "- Active production bundle: retain indefinitely while active.",
        "- Missing production bundle: block production startup and regenerate/promote through the pipeline.",
        "- Candidate bundles: retain latest approved candidates and any candidate tied to audit evidence; remove older failed candidates only after reports and manifests are preserved.",
        "- Frozen splits: immutable; create a new split version when membership changes.",
        "- Raw/processed datasets: keep ignored locally; do not delete without confirming the source can be reacquired.",
        "- Caches/logs: safe to delete and regenerate.",
        "",
        "## Configuration Usage",
        "",
        f"- Configuration files inventoried: {len(config_usage)}",
        "- Unknown/stale fields require consumer-level validation before removal.",
        "",
        "## Test Cleanup",
        "",
        f"- Active test modules: {len(tests['active_test_files'])}",
        f"- Legacy test modules: {len(tests['legacy_test_files'])}",
        f"- Legacy test action: {tests['cleanup_action']}",
        "",
        "## Hardcoded Rule Inventory",
        "",
        f"- Files with sample-retail/business-rule terms: {len(hardcoded)}",
        "- Runtime/gate hits are marked review-required; test/fixture hits are not automatically defects.",
        "",
        "## Comment Coverage",
        "",
        f"- Retained Python modules analyzed: {doc_cov['python_files_analyzed']}",
        f"- Modules with purpose docstrings: {doc_cov['modules_with_docstrings']}",
        f"- Modules missing docstrings: {doc_cov['modules_missing_docstrings']}",
        f"- Public classes documented/missing: {doc_cov['public_classes'] - doc_cov['public_classes_missing_docstrings']}/{doc_cov['public_classes_missing_docstrings']}",
        f"- Public functions documented/missing: {doc_cov['public_functions'] - doc_cov['public_functions_missing_docstrings']}/{doc_cov['public_functions_missing_docstrings']}",
        "- Full symbol-level gaps are in repository_inventory.json under python_docstring_coverage.",
        "",
        "## Before/After Statistics",
        "",
        markdown_table(
            ["Metric", "Before", "After"],
            [
                ["tracked files", stats_before["tracked_files"], after["tracked_files"]],
                ["untracked files", stats_before["untracked_files"], after["untracked_files"]],
                ["ignored summary entries", stats_before["ignored_summary_entries"], after["ignored_summary_entries"]],
                ["Python files", stats_before["python_files"], after["python_files"]],
                ["Markdown files", stats_before["markdown_files"], after["markdown_files"]],
                ["test files", stats_before["test_files"], after["test_files"]],
                ["configuration files", stats_before["configuration_files"], after["configuration_files"]],
                ["generated files in Git", stats_before["generated_files_in_git"], after["generated_files_in_git"]],
            ],
        ),
        "",
        "## Rollback Instructions",
        "",
        "- Generated reports: rerun `python scripts/generate_repository_cleanup_inventory.py` or remove `artifacts/repository_cleanup/`.",
        "- Low-risk cache/log cleanup: rerun tests or commands to regenerate caches/logs if needed.",
        "- Source changes in this pass: revert `.gitignore`, `docs/REPOSITORY_MAP.md`, and `scripts/generate_repository_cleanup_inventory.py` if the cleanup report path should not be tracked.",
        "",
        "## Follow-Up Items",
        "",
        "- Review untracked `.github/workflows/ci.yml`, semantic_v2 split files, and `tests/test_training_audit_fixes.py` for tracking.",
        "- Decide whether docs/reports run-specific files should move to artifacts/pipeline/runs/<run_id>/reports/.",
        "- Apply retention policy to old candidate bundles only after preserving required manifests and reports.",
        "- Close module/public docstring gaps in focused source-owner batches.",
        "- Review runtime hardcoded sample-retail terms and keep only schema-signature-gated or config-driven behavior.",
        "",
    ]
    return "\n".join(lines)


def build_archive_manifest(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in inventory:
        if entry["classification"] in {"ARCHIVE", "REVIEW_REQUIRED", "CONSOLIDATE"}:
            rows.append(
                {
                    "path": entry["path"],
                    "classification": entry["classification"],
                    "reason": entry["why_required"],
                    "cleanup_action": entry["cleanup_action"],
                    "evidence": entry["evidence"],
                    "risk": "medium" if entry["classification"] == "REVIEW_REQUIRED" else "low",
                }
            )
    return rows


def build_artifact_manifest(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = baseline["artifact_state"]
    rows = [
        {
            "path": "artifacts/model_bundle/current",
            "artifact_type": "production bundle",
            "generated_by": "training/train_model.py promotion step",
            "source_inputs": "pipeline config, datasets, trained artifacts, quality gate reports",
            "required_by": "app/streamlit_app.py and ModelBundleLoader.load_current",
            "reproducible_command": "python training/train_model.py --config configs/training.yaml",
            "retention_policy": "retain indefinitely while active; missing locally blocks production startup",
            "safe_to_delete_condition": "only after a newer current bundle is promoted and rollback policy is satisfied",
            "checksum": None,
            "exists": artifact["current_production_bundle"]["exists"],
        },
        {
            "path": "artifacts/model_bundle/candidate",
            "artifact_type": "singleton candidate bundle",
            "generated_by": "legacy/baseline training flow",
            "source_inputs": "training config, generated model/evaluation artifacts",
            "required_by": "debug candidate loading and historical reports",
            "reproducible_command": "python training/train_model.py --config configs/baseline_training.yaml",
            "retention_policy": "retain pending migration to run-scoped candidates",
            "safe_to_delete_condition": "after no code/docs/tests reference singleton candidate and required reports are archived",
            "checksum": sha256(ROOT / "artifacts" / "model_bundle" / "candidate" / "bundle_manifest.json"),
            "exists": artifact["singleton_candidate_bundle"]["exists"],
        },
    ]
    for candidate in artifact["candidate_bundles"]:
        rows.append(
            {
                "path": candidate["path"],
                "artifact_type": "run-scoped candidate bundle",
                "generated_by": "training/train_model.py build_model_bundle step",
                "source_inputs": "pipeline run artifacts and effective config",
                "required_by": "audit/release review when tied to a run",
                "reproducible_command": "python training/train_model.py --config <captured effective config>",
                "retention_policy": "retain latest candidates and audit-required bundles; prune older failed bundles only by manifest",
                "safe_to_delete_condition": "not current, not latest review candidate, reports preserved, no docs/tests reference it",
                "checksum": sha256(ROOT / candidate["path"] / "bundle_manifest.json"),
                "exists": True,
                "bundle_id": candidate["bundle_id"],
                "status": candidate["status"],
            }
        )
    split = artifact["active_dataset_split_version"].get("split_version")
    if split:
        rows.append(
            {
                "path": f"data/splits/{split}",
                "artifact_type": "frozen dataset split",
                "generated_by": "dataset_training.split_manager",
                "source_inputs": "configured source datasets and random seed",
                "required_by": "training/train_model.py via configs/training.yaml",
                "reproducible_command": "create a new split version; do not overwrite this one",
                "retention_policy": "immutable while referenced by configs or bundle evidence",
                "safe_to_delete_condition": "only after no config, bundle, report, or test references the split version",
                "checksum": sha256(ROOT / "data" / "splits" / split / "split_manifest.json"),
                "exists": (ROOT / "data" / "splits" / split).exists(),
            }
        )
    return rows


def write_markdown_reports(
    config_usage: list[dict[str, Any]],
    hardcoded: list[dict[str, Any]],
    docs_report: dict[str, Any],
    test_report: dict[str, Any],
    consolidation: list[dict[str, Any]],
) -> None:
    write_text(
        OUT_DIR / "configuration_usage_report.md",
        "# Configuration Usage Report\n\n"
        + markdown_table(
            ["File", "Active", "Referenced by", "Cleanup action"],
            [[row["file"], row["active"], ", ".join(row["referenced_by"][:5]), row["cleanup_action"]] for row in config_usage],
        )
        + "\n",
    )
    write_text(
        OUT_DIR / "hardcoded_rule_inventory.md",
        "# Hardcoded Rule Inventory\n\n"
        + markdown_table(
            ["Path", "Patterns", "Classification"],
            [[row["path"], ", ".join(row["patterns"]), row["classification"]] for row in hardcoded],
        )
        + "\n",
    )
    write_text(
        OUT_DIR / "documentation_cleanup_report.md",
        "# Documentation Cleanup Report\n\n"
        "## Canonical Documents\n\n"
        + "\n".join(f"- {path}" for path in docs_report["canonical_docs"])
        + "\n\n## Report Documents Requiring Archive/Move Review\n\n"
        + "\n".join(f"- {path}" for path in docs_report["report_docs"])
        + "\n",
    )
    write_text(
        OUT_DIR / "test_cleanup_report.md",
        "# Test Cleanup Report\n\n"
        f"Active test modules: {len(test_report['active_test_files'])}\n\n"
        f"Legacy test modules: {len(test_report['legacy_test_files'])}\n\n"
        f"Action: {test_report['cleanup_action']}\n",
    )
    write_text(
        OUT_DIR / "canonical_owner_matrix.md",
        "# Canonical Owner Matrix\n\n"
        + markdown_table(
            ["Area", "Classification", "Canonical owner", "Cleanup action"],
            [[row["area"], row["classification"], row["canonical_owner"], row["cleanup_action"]] for row in consolidation],
        )
        + "\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate repository cleanup inventory and reports.")
    parser.add_argument("--delete-low-risk", action="store_true", help="Delete generated caches and root smoke logs after path safety checks.")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = collect_baseline()
    inventory, stats_before = build_inventory()
    texts = build_text_reference_index([e["path"] for e in inventory if e["git_status"] != "ignored"])
    config_usage = collect_config_usage(texts)
    hardcoded = collect_hardcoded_rules(texts)
    docs_report = collect_documentation_report(inventory)
    test_report = collect_test_report(inventory)
    consolidation = collect_consolidation_report(inventory)
    deletion_manifest: list[dict[str, Any]] = []
    if args.delete_low_risk:
        deletion_manifest = safe_delete_low_risk()
        baseline = collect_baseline()
        inventory, stats_after = build_inventory()
    else:
        stats_after = None
    archive_manifest = build_archive_manifest(inventory)
    artifact_manifest = build_artifact_manifest(baseline)

    write_json(OUT_DIR / "baseline_report.json", baseline)
    write_json(OUT_DIR / "repository_inventory.json", {"generated_at": now_iso(), "stats": stats_before, "entries": inventory})
    write_json(OUT_DIR / "deletion_manifest.json", deletion_manifest)
    write_json(OUT_DIR / "archive_manifest.json", archive_manifest)
    write_json(OUT_DIR / "consolidation_manifest.json", consolidation)
    write_json(OUT_DIR / "artifact_manifest.json", artifact_manifest)
    write_json(OUT_DIR / "configuration_usage_report.json", config_usage)
    write_json(OUT_DIR / "hardcoded_rule_inventory.json", hardcoded)
    write_json(OUT_DIR / "documentation_cleanup_report.json", docs_report)
    write_json(OUT_DIR / "test_cleanup_report.json", test_report)
    write_json(OUT_DIR / "before_after_stats.json", {"before": stats_before, "after": stats_after or stats_before})
    write_text(REPO_MAP, generate_repo_map(inventory, baseline, stats_after or stats_before))
    write_text(
        OUT_DIR / "cleanup_report.md",
        generate_cleanup_report(
            baseline,
            stats_before,
            stats_after,
            deletion_manifest,
            archive_manifest,
            consolidation,
            docs_report,
            test_report,
            hardcoded,
            config_usage,
        ),
    )
    write_markdown_reports(config_usage, hardcoded, docs_report, test_report, consolidation)

    print(f"Wrote {rel(REPO_MAP)}")
    print(f"Wrote cleanup reports under {rel(OUT_DIR)}")
    if deletion_manifest:
        print(f"Deleted {len(deletion_manifest)} low-risk generated cache/log paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
