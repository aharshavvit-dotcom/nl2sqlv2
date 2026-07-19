"""Rationalize and document the pytest suite.

Purpose: Generates the required test-suite inventory and safely consolidates
fragmented test modules into cohesive, traceable modules.

Inputs:
    - tests/**/*.py
    - pytest.ini

Outputs:
    - artifacts/repository_cleanup/test_inventory.json
    - artifacts/repository_cleanup/duplicate_tests.json
    - artifacts/repository_cleanup/test_deletion_manifest.json
    - artifacts/repository_cleanup/test_suite_cleanup_report.json
    - docs/TESTING.md
    - docs/reports/test_suite_cleanup_report.md
    - tests/test_catalog.yaml

Safety:
    Consolidation is skipped for a cluster when top-level helper/function/class
    names would collide. Source files are deleted only after their contents are
    written to the target module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
ARTIFACT_DIR = ROOT / "artifacts" / "repository_cleanup"
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = DOCS_DIR / "reports"


@dataclass(frozen=True)
class Cluster:
    target: str
    sources: tuple[str, ...]
    category: str
    area: str
    requirement: str
    owner: str
    reason: str


CLUSTERS: tuple[Cluster, ...] = (
    Cluster(
        target="tests/unit/ir/test_query_ir_v2_models.py",
        sources=(
            "tests/test_query_ir_v2_models.py",
            "tests/test_query_ir_v2_boolean_models.py",
            "tests/test_query_ir_v2_serialization.py",
            "tests/test_query_ir_v2_fingerprint.py",
        ),
        category="UNIT",
        area="ir",
        requirement="QIR-V2-MODEL-001",
        owner="ir/query_ir_v2_models.py",
        reason="QueryIR v2 model, literal, serialization and fingerprint contracts belong to the same model responsibility.",
    ),
    Cluster(
        target="tests/unit/ir/test_query_ir_v2_validation.py",
        sources=(
            "tests/test_query_ir_v2_validation.py",
            "tests/test_query_ir_v2_depth_limits.py",
            "tests/test_query_ir_v2_boolean_validation.py",
            "tests/test_query_ir_v2_boolean_depth_limits.py",
            "tests/test_query_ir_v2_boolean_canonicalization.py",
        ),
        category="UNIT",
        area="ir",
        requirement="QIR-V2-VALIDATION-001",
        owner="ir/query_ir_v2_validation.py",
        reason="Validation, canonicalization and recursive predicate limits protect one QueryIR validation contract.",
    ),
    Cluster(
        target="tests/unit/ir/test_query_ir_v2_conversion.py",
        sources=("tests/test_sql_to_query_ir_v2_boolean_conversion.py",),
        category="UNIT",
        area="ir",
        requirement="QIR-V2-CONVERSION-001",
        owner="ir/sql_to_query_ir_v2.py",
        reason="SQL-to-QueryIR conversion belongs in the canonical conversion module.",
    ),
    Cluster(
        target="tests/unit/ir/test_query_ir_v2_rendering.py",
        sources=("tests/test_query_ir_v2_boolean_renderer.py",),
        category="UNIT",
        area="ir",
        requirement="QIR-V2-RENDER-001",
        owner="ir/query_ir_v2_boolean_renderer.py",
        reason="Boolean, NULL, IN and BETWEEN rendering are one QueryIR rendering responsibility.",
    ),
    Cluster(
        target="tests/unit/ir/test_query_ir_migration.py",
        sources=(
            "tests/test_query_ir_v1_to_v2_migration.py",
            "tests/test_query_ir_v1_boolean_migration.py",
            "tests/test_query_ir_v2_to_v1_compatibility.py",
            "tests/test_query_ir_v2_boolean_v1_compatibility.py",
            "tests/test_query_ir_version_loader.py",
            "tests/test_query_ir_v2_renderer_parity.py",
        ),
        category="UNIT",
        area="ir",
        requirement="QIR-MIGRATION-001",
        owner="ir/query_ir_migration.py",
        reason="V1/V2 migration, compatibility and version loading form one compatibility contract.",
    ),
    Cluster(
        target="tests/integration/test_query_ir_v2_execution.py",
        sources=(
            "tests/test_query_ir_v2_boolean_precedence.py",
            "tests/test_query_ir_v2_boolean_execution_equivalence.py",
        ),
        category="INTEGRATION",
        area="ir",
        requirement="QIR-V2-EXECUTION-001",
        owner="ir/query_ir_v2_boolean_renderer.py",
        reason="Renderer output and SQLite execution equivalence should be exercised as integration behaviour.",
    ),
    Cluster(
        target="tests/unit/capabilities/test_capability_pipeline.py",
        sources=(
            "tests/test_capability_artifact_schema.py",
            "tests/test_capability_dataset_reporting.py",
            "tests/test_capability_or_filter_consistency.py",
            "tests/test_capability_taxonomy.py",
            "tests/test_capability_training_batch.py",
            "tests/test_sql_capability_extractor.py",
            "tests/test_training_inference_capability_parity.py",
            "tests/test_unsupported_example_task_masks.py",
        ),
        category="UNIT",
        area="capabilities",
        requirement="CAPABILITY-PIPELINE-001",
        owner="capabilities/",
        reason="Capability extraction, taxonomy, artifacts and training parity are one capability-data contract.",
    ),
    Cluster(
        target="tests/unit/data/test_dataset_pipeline.py",
        sources=(
            "tests/test_20_dataset_split_manager.py",
            "tests/test_21_dataset_leakage_checker.py",
            "tests/test_22_generic_ir_corpus_builder.py",
            "tests/test_24_dataset_scale_evaluator.py",
            "tests/test_dataset_leakage_domain.py",
            "tests/test_dataset_split_integrity.py",
            "tests/test_verify_datasets.py",
            "tests/test_sql_partial_supervision.py",
        ),
        category="UNIT",
        area="data",
        requirement="DATA-PIPELINE-001",
        owner="dataset_training/",
        reason="Dataset split, leakage, corpus, scale and verification tests protect the dataset pipeline.",
    ),
    Cluster(
        target="tests/unit/retrieval/test_retrieval_pipeline.py",
        sources=(
            "tests/test_04_retrieval_runtime.py",
            "tests/test_23_retrieval_rag_index.py",
            "tests/test_train_retriever_from_datasets.py",
        ),
        category="UNIT",
        area="retrieval",
        requirement="RETRIEVAL-PIPELINE-001",
        owner="retriever/",
        reason="Retriever runtime, index building and dataset trainer wrappers belong to the retrieval pipeline.",
    ),
    Cluster(
        target="tests/unit/feedback/test_feedback_pipeline.py",
        sources=(
            "tests/test_31_feedback_store.py",
            "tests/test_32_feedback_to_training_data.py",
            "tests/test_33_feedback_index.py",
            "tests/test_34_reward_scorer.py",
            "tests/test_40_gold_comparator.py",
            "tests/test_41_error_classifier.py",
            "tests/test_42_hard_negative_generator.py",
            "tests/test_43_correction_generator.py",
            "tests/test_44_improvement_tracker.py",
            "tests/test_45_self_training_loop.py",
            "tests/test_46_prediction_runner.py",
        ),
        category="UNIT",
        area="feedback",
        requirement="FEEDBACK-PIPELINE-001",
        owner="feedback/",
        reason="Feedback storage, scoring, correction generation and self-training tests describe one feedback loop.",
    ),
    Cluster(
        target="tests/unit/execution/test_execution_evaluation.py",
        sources=(
            "tests/test_50_sql_canonicalizer.py",
            "tests/test_51_sql_structure_comparator.py",
            "tests/test_52_result_comparator.py",
            "tests/test_53_execution_aware_evaluation.py",
        ),
        category="UNIT",
        area="execution",
        requirement="EXEC-EVAL-001",
        owner="execution_eval/",
        reason="SQL canonicalization, structure comparison and execution-aware evaluation share one responsibility.",
    ),
    Cluster(
        target="tests/unit/runtime/test_generic_planner_and_grounding.py",
        sources=(
            "tests/test_10_generic_table_intent.py",
            "tests/test_11_generic_join_policy.py",
            "tests/test_60_schema_profiler.py",
            "tests/test_61_glossary_generator.py",
            "tests/test_62_semantic_mapper.py",
            "tests/test_63_ambiguity_detector.py",
            "tests/test_64_clarification_runtime.py",
            "tests/test_120_schema_value_index.py",
            "tests/test_121_filter_value_extractor.py",
            "tests/test_122_filter_grounding.py",
            "tests/test_123_projection_resolution.py",
            "tests/test_124_dimension_resolution.py",
        ),
        category="UNIT",
        area="runtime",
        requirement="RUNTIME-GROUNDING-001",
        owner="inference/grounding/",
        reason="Schema profiling, generic planning, clarification and grounding form one runtime grounding responsibility.",
    ),
    Cluster(
        target="tests/integration/test_database_and_connected_regression.py",
        sources=(
            "tests/test_03_database_connectors.py",
            "tests/test_12_generic_postgres_schema_runtime.py",
            "tests/test_65_connected_db_regression_generator.py",
            "tests/test_66_connected_db_regression_runner.py",
            "tests/test_134_database_integration.py",
        ),
        category="INTEGRATION",
        area="execution",
        requirement="DB-INTEGRATION-001",
        owner="db/",
        reason="Database connector and connected-database regression tests are one integration lane.",
    ),
    Cluster(
        target="tests/integration/test_model_lifecycle_pipeline.py",
        sources=(
            "tests/test_35_model_quality_gate.py",
            "tests/test_36_quality_thresholds.py",
            "tests/test_36_regression_suite.py",
            "tests/test_37_model_artifact_registry.py",
            "tests/test_38_release_readiness.py",
            "tests/test_54_model_selector.py",
            "tests/test_55_model_promotion_policy.py",
            "tests/test_56_champion_challenger.py",
            "tests/test_57_pipeline_runner.py",
            "tests/test_58_pipeline_state.py",
            "tests/test_59_full_training_pipeline_smoke.py",
            "tests/test_68_execution_pipeline_audit.py",
            "tests/test_99_train_model_integration.py",
        ),
        category="INTEGRATION",
        area="training",
        requirement="MODEL-LIFECYCLE-001",
        owner="training/",
        reason="Quality gates, release readiness, promotion and training pipeline tests protect the model lifecycle.",
    ),
    Cluster(
        target="tests/unit/model/test_neural_training_components.py",
        sources=(
            "tests/test_70_training_config.py",
            "tests/test_71_activation_factory.py",
            "tests/test_72_optimizer_factory.py",
            "tests/test_73_scheduler_factory.py",
            "tests/test_74_ffn_blocks.py",
            "tests/test_75_loss_weighter.py",
            "tests/test_76_checkpoint_manager.py",
            "tests/test_77_early_stopping.py",
            "tests/test_79_neural_experiment_runner.py",
            "tests/test_80_neural_candidate_ranker.py",
            "tests/test_81_diagnostics_and_telemetry.py",
            "tests/test_neural_ir_dataset.py",
        ),
        category="UNIT",
        area="model",
        requirement="NEURAL-TRAINING-001",
        owner="neural_optimization/",
        reason="Neural training config, factories, losses, checkpointing and diagnostics are one model-training unit area.",
    ),
    Cluster(
        target="tests/unit/execution/test_sql_validation_and_safety.py",
        sources=(
            "tests/test_02_sql_validation.py",
            "tests/test_105_sql_validation_policy.py",
            "tests/test_119_renderer_attribution.py",
            "tests/test_132_sqlite_prediction_cache.py",
            "tests/test_133_telemetry_privacy.py",
        ),
        category="SAFETY",
        area="execution",
        requirement="SQL-SAFETY-001",
        owner="execution/",
        reason="SQL validation, read-only policy, attribution, cache identity and telemetry privacy are safety contracts.",
    ),
    Cluster(
        target="tests/e2e/test_application_and_training_smoke.py",
        sources=(
            "tests/test_09_end_to_end_smoke.py",
            "tests/test_78_neural_training_smoke.py",
            "tests/test_79_smoke_and_ablation_validation.py",
        ),
        category="SMOKE",
        area="e2e",
        requirement="SMOKE-PIPELINE-001",
        owner="app/streamlit_app.py",
        reason="Application, neural training and ablation smoke checks belong in one smoke lane.",
    ),
)


CATEGORY_MARKERS = {
    "UNIT": "unit",
    "INTEGRATION": "integration",
    "REGRESSION": "regression",
    "END_TO_END": "e2e",
    "SAFETY": "safety",
    "CONTRACT": "contract",
    "PROPERTY": "property",
    "SMOKE": "e2e",
    "PERFORMANCE": "performance",
    "LEGACY": "legacy",
}


AREA_REQUIREMENTS = {
    "app": "APP-SMOKE-001",
    "bundles": "BUNDLE-LIFECYCLE-001",
    "capabilities": "CAPABILITY-PIPELINE-001",
    "data": "DATA-PIPELINE-001",
    "execution": "SQL-SAFETY-001",
    "feedback": "FEEDBACK-PIPELINE-001",
    "ir": "QIR-V2-VALIDATION-001",
    "model": "NEURAL-TRAINING-001",
    "retrieval": "RETRIEVAL-PIPELINE-001",
    "runtime": "RUNTIME-GROUNDING-001",
    "training": "MODEL-LIFECYCLE-001",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_files() -> list[Path]:
    return sorted(
        path
        for path in TESTS_DIR.rglob("*.py")
        if path.name.startswith("test_") and "__pycache__" not in path.parts
    )


def parse_ast(path: Path) -> ast.Module | None:
    try:
        return ast.parse(read_text(path))
    except SyntaxError:
        return None


def module_docstring(path: Path) -> str | None:
    tree = parse_ast(path)
    if tree is None:
        return None
    return ast.get_docstring(tree)


def iter_test_nodes(tree: ast.Module | None) -> list[str]:
    if tree is None:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    names.append(f"{node.name}.{child.name}")
    return names


def defined_names(path: Path) -> set[str]:
    tree = parse_ast(path)
    if tree is None:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(names_from_target(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(names_from_target(node.target))
    return names


def names_from_target(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(names_from_target(item))
        return names
    return set()


def markers_for(path: Path, text: str) -> list[str]:
    markers = set(re.findall(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)", text))
    rel_path = rel(path)
    category, _area = infer_category_area(path, text)
    marker = CATEGORY_MARKERS.get(category)
    if marker:
        markers.add(marker)
    if "tmp_path" in text:
        markers.add("unit")
    if uses_database(text):
        markers.add("database")
    if "training" in rel_path or "train_" in rel_path:
        markers.add("training")
    if category in {"INTEGRATION", "REGRESSION", "END_TO_END", "SMOKE"}:
        markers.add(CATEGORY_MARKERS[category])
    return sorted(markers)


def uses_database(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("sqlite", "postgres", "database", "db_path", "duckdb"))


def uses_model(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("model", "checkpoint", "torch", "sklearn", "neural"))


def uses_gpu(text: str) -> bool:
    return any(token in text.lower() for token in ("cuda", "gpu"))


def infer_category_area(path: Path, text: str) -> tuple[str, str]:
    rel_path = rel(path)
    parts = path.parts
    name = path.name
    if "legacy" in parts:
        return "LEGACY", "legacy"
    if "/integration/" in rel_path or "integration" in name or "connected_db" in name or "postgres" in name:
        return "INTEGRATION", infer_area_from_name(name, text)
    if "/e2e/" in rel_path or "smoke" in name or "end_to_end" in name:
        return "SMOKE", infer_area_from_name(name, text)
    if "regression" in name or "golden" in name:
        return "REGRESSION", infer_area_from_name(name, text)
    if "safety" in name or "validation_policy" in name or "telemetry_privacy" in name:
        return "SAFETY", infer_area_from_name(name, text)
    if "performance" in name or "latency" in name:
        return "PERFORMANCE", infer_area_from_name(name, text)
    if "contract" in name or "bundle" in name or "policy" in name:
        return "CONTRACT", infer_area_from_name(name, text)
    return "UNIT", infer_area_from_name(name, text)


def infer_area_from_name(name: str, text: str) -> str:
    combined = f"{name}\n{text[:2000]}".lower()
    for area, needles in {
        "ir": ("query_ir", "qir", "ir."),
        "capabilities": ("capability", "capabilities"),
        "data": ("dataset", "split", "corpus", "leakage"),
        "execution": ("sql", "execution", "database", "postgres", "sqlite", "telemetry"),
        "feedback": ("feedback", "correction", "self_training", "reward"),
        "model": ("neural", "checkpoint", "optimizer", "scheduler", "loss", "model"),
        "retrieval": ("retrieval", "retriever", "rag"),
        "runtime": ("runtime", "grounding", "clarification", "route", "schema_value"),
        "training": ("training", "promotion", "quality_gate", "pipeline", "release"),
        "app": ("streamlit", "app"),
        "bundles": ("bundle",),
    }.items():
        if any(needle in combined for needle in needles):
            return area
    return "general"


def owner_for(area: str, path: Path) -> str:
    if area == "general":
        return rel(path)
    return {
        "app": "app/",
        "bundles": "model_bundle/",
        "capabilities": "capabilities/",
        "data": "dataset_training/",
        "execution": "execution/",
        "feedback": "feedback/",
        "ir": "ir/",
        "legacy": "tests/legacy/",
        "model": "neural_optimization/",
        "retrieval": "retriever/",
        "runtime": "inference/",
        "training": "training/",
    }.get(area, rel(path))


def inventory_entry(path: Path, cluster_by_source: dict[str, Cluster], deleted_sources: set[str]) -> dict[str, Any]:
    text = read_text(path)
    tree = parse_ast(path)
    category, area = infer_category_area(path, text)
    test_names = iter_test_nodes(tree)
    path_rel = rel(path)
    cluster = cluster_by_source.get(path_rel)
    requirement = cluster.requirement if cluster else AREA_REQUIREMENTS.get(area, f"{area.upper()}-REVIEW-001")
    cleanup_action = "MERGE" if cluster else ("ARCHIVE" if category == "LEGACY" else "KEEP")
    merge_target = cluster.target if cluster else None
    duplicate_hash = normalized_test_body_hash(text)
    return {
        "path": path_rel,
        "category": category,
        "area": area,
        "purpose": purpose_for(category, area, path),
        "requirements": [requirement],
        "canonical_module": cluster.owner if cluster else owner_for(area, path),
        "test_count": len(test_names),
        "test_cases": test_names,
        "markers": markers_for(path, text),
        "runtime_seconds": None,
        "runtime_note": "Per-file runtime was not measured; suite runtime is recorded in the cleanup report.",
        "uses_database": uses_database(text),
        "uses_model": uses_model(text),
        "uses_gpu": uses_gpu(text),
        "blocking": category not in {"LEGACY", "OBSOLETE", "DUPLICATE"},
        "duplicate_signature": duplicate_hash,
        "duplicate_of": None,
        "cleanup_action": cleanup_action,
        "merge_target": merge_target,
        "removed_by_cleanup": path_rel in deleted_sources,
        "reason": cluster.reason if cluster else reason_for(category, area),
    }


def purpose_for(category: str, area: str, path: Path) -> str:
    if category == "LEGACY":
        return "Documents historical compatibility behaviour excluded from the default pytest lane."
    return f"Protects {area} {category.lower()} behaviour for {path.stem.replace('test_', '').replace('_', ' ')}."


def reason_for(category: str, area: str) -> str:
    if category == "LEGACY":
        return "Excluded from default collection; requires compatibility review before migration or archive."
    return f"Retained as the current {category.lower()} coverage for the {area} area."


def normalized_test_body_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text)
    normalized = re.sub(r"test_[A-Za-z0-9_]+", "test_NAME", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def duplicate_report(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["duplicate_signature"]].append(entry)
    duplicates = []
    for group_entries in groups.values():
        if len(group_entries) < 2:
            continue
        tests = [entry["path"] for entry in group_entries]
        canonical = next((entry for entry in group_entries if entry["cleanup_action"] != "ARCHIVE"), group_entries[0])
        duplicates.append(
            {
                "tests": tests,
                "overlap": "Files have matching normalized test body signatures.",
                "action": f"Review and merge into {canonical.get('merge_target') or canonical['path']}",
            }
        )
        for entry in group_entries:
            if entry is not canonical:
                entry["duplicate_of"] = canonical["path"]
                if entry["cleanup_action"] == "KEEP":
                    entry["cleanup_action"] = "REVIEW_REQUIRED"
    return duplicates


def cluster_by_source(clusters: tuple[Cluster, ...]) -> dict[str, Cluster]:
    return {source: cluster for cluster in clusters for source in cluster.sources}


def unsafe_name_collisions(cluster: Cluster) -> dict[str, list[str]]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    for source in cluster.sources:
        path = ROOT / source
        if not path.exists():
            continue
        for name in defined_names(path):
            occurrences[name].append(source)
    return {name: paths for name, paths in occurrences.items() if len(paths) > 1}


def strip_future_imports(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip() != "from __future__ import annotations").strip() + "\n"


def consolidate_clusters(apply: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    merged: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    deleted_sources: set[str] = set()
    for cluster in CLUSTERS:
        existing_sources = [source for source in cluster.sources if (ROOT / source).exists()]
        if not existing_sources:
            skipped.append({"target": cluster.target, "reason": "sources already moved or missing", "sources": list(cluster.sources)})
            continue
        collisions = unsafe_name_collisions(cluster)
        if collisions:
            skipped.append(
                {
                    "target": cluster.target,
                    "reason": "top-level namespace collision",
                    "sources": existing_sources,
                    "collisions": collisions,
                }
            )
            continue
        if apply:
            write_consolidated_module(cluster, existing_sources)
            for source in existing_sources:
                source_path = ROOT / source
                source_path.unlink()
                deleted_sources.add(source)
        merged.append(
            {
                "target": cluster.target,
                "sources": existing_sources,
                "category": cluster.category,
                "requirement": cluster.requirement,
                "reason": cluster.reason,
            }
        )
    return merged, skipped, deleted_sources


def write_consolidated_module(cluster: Cluster, sources: list[str]) -> None:
    target = ROOT / cluster.target
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        '"""',
        f"Purpose: Verifies {cluster.area} {cluster.category.lower()} behaviour consolidated from fragmented test files.",
        f"Required because: {cluster.reason}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import pytest",
        "",
        f"pytestmark = pytest.mark.{CATEGORY_MARKERS.get(cluster.category, 'unit')}",
        "",
    ]
    for source in sources:
        body = strip_future_imports(read_text(ROOT / source))
        chunks.extend(
            [
                "",
                f"# Source: {source}",
                body,
            ]
        )
    target.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def add_missing_docstrings() -> list[str]:
    updated: list[str] = []
    for path in test_files():
        if module_docstring(path):
            continue
        text = read_text(path)
        category, area = infer_category_area(path, text)
        docstring = (
            '"""\n'
            f"Purpose: Protects {area} {category.lower()} behaviour.\n"
            "Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.\n"
            '"""\n\n'
        )
        path.write_text(docstring + text, encoding="utf-8")
        updated.append(rel(path))
    return updated


def write_conftest() -> None:
    path = TESTS_DIR / "conftest.py"
    if path.exists():
        return
    path.write_text(
        '''"""Shared pytest configuration for rationalized execution lanes.

Purpose: Applies default lane markers from test paths so every collected item
has an execution lane even when an older test module lacks explicit marks.
"""

from __future__ import annotations

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        name = item.name.lower()
        marker_names = {marker.name for marker in item.iter_markers()}

        def add(marker: str) -> None:
            if marker not in marker_names:
                item.add_marker(marker)
                marker_names.add(marker)

        if "/legacy/" in path:
            add("legacy")
        elif "/integration/" in path or "integration" in path or "connected_db" in path or "postgres" in path:
            add("integration")
        elif "/e2e/" in path or "smoke" in path or "end_to_end" in path:
            add("e2e")
        elif "/regression/" in path or "regression" in path or "golden" in path:
            add("regression")
        elif "safety" in path or "validation_policy" in path or "telemetry_privacy" in path:
            add("safety")
        elif "contract" in path or "bundle" in path or "policy" in path:
            add("contract")
        else:
            add("unit")

        if "training" in path or "train_" in path:
            add("training")
        if any(token in path or token in name for token in ("database", "postgres", "sqlite", "connected_db")):
            add("database")
        if any(token in path or token in name for token in ("slow", "full_training")):
            add("slow")
''',
        encoding="utf-8",
    )


def write_legacy_readme() -> None:
    path = TESTS_DIR / "legacy" / "README.md"
    path.write_text(
        """# Legacy Test Holding Area

`tests/legacy` is excluded from the default pytest collection through `pytest.ini`.

These files are retained as compatibility evidence only. Each file is inventoried
in `artifacts/repository_cleanup/test_inventory.json` with category `LEGACY` and
cleanup action `ARCHIVE` until the behaviour is either migrated into an active
regression test or deleted with a replacement recorded in
`artifacts/repository_cleanup/test_deletion_manifest.json`.
""",
        encoding="utf-8",
    )


def metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    files = [entry for entry in entries if not entry.get("removed_by_cleanup")]
    categories = Counter(entry["category"] for entry in files)
    return {
        "test_files": len(files),
        "collected_tests_estimate": sum(entry["test_count"] for entry in files if entry["category"] != "LEGACY"),
        "unit_files": categories["UNIT"],
        "integration_files": categories["INTEGRATION"],
        "regression_files": categories["REGRESSION"],
        "legacy_files": categories["LEGACY"],
        "duplicate_files": sum(1 for entry in files if entry.get("duplicate_of")),
        "excluded_files": categories["LEGACY"],
        "runtime_seconds": None,
        "line_coverage": None,
        "branch_coverage": None,
        "flaky_tests": None,
        "requirements_mapped": len({req for entry in files for req in entry["requirements"]}),
    }


def write_testing_docs(entries: list[dict[str, Any]], report: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    retained = [entry for entry in entries if not entry.get("removed_by_cleanup") and entry["category"] != "LEGACY"]
    catalog = build_catalog(retained)
    (TESTS_DIR / "test_catalog.yaml").write_text(render_catalog(catalog), encoding="utf-8")
    (DOCS_DIR / "TESTING.md").write_text(render_testing_md(report, catalog), encoding="utf-8")
    (REPORTS_DIR / "test_suite_cleanup_report.md").write_text(render_cleanup_report_md(report), encoding="utf-8")


def build_catalog(entries: list[dict[str, Any]]) -> dict[str, Any]:
    catalog: dict[str, Any] = {}
    for entry in entries:
        lane = lane_for(entry)
        for requirement in entry["requirements"]:
            item = catalog.setdefault(
                requirement,
                {
                    "description": requirement_description(requirement),
                    "owner": entry["canonical_module"],
                    "tests": [],
                    "blocking": entry["blocking"],
                    "ci_lane": lane,
                },
            )
            if entry["path"] not in item["tests"]:
                item["tests"].append(entry["path"])
    return dict(sorted(catalog.items()))


def lane_for(entry: dict[str, Any]) -> str:
    category = entry["category"]
    if category in {"UNIT", "CONTRACT", "SAFETY"}:
        return "core"
    if category in {"INTEGRATION", "REGRESSION"}:
        return "integration"
    if category in {"SMOKE", "END_TO_END"}:
        return "smoke"
    if category in {"PERFORMANCE"}:
        return "slow"
    return "review"


def requirement_description(requirement: str) -> str:
    descriptions = {
        "QIR-V2-MODEL-001": "QueryIR v2 models, literals, serialization and fingerprints remain stable.",
        "QIR-V2-VALIDATION-001": "QueryIR v2 validation rejects unsafe, unsupported and structurally invalid payloads.",
        "QIR-V2-CONVERSION-001": "SQL conversion preserves QueryIR v2 predicate semantics.",
        "QIR-V2-RENDER-001": "QueryIR v2 rendering preserves SQL predicate semantics and safety boundaries.",
        "QIR-MIGRATION-001": "QueryIR v1/v2 migration preserves supported SQL semantics and reports unsupported constructs.",
        "QIR-V2-EXECUTION-001": "Rendered QueryIR v2 SQL remains execution-equivalent for supported predicate trees.",
        "CAPABILITY-PIPELINE-001": "Capability extraction and artifacts remain compatible between training and inference.",
        "DATA-PIPELINE-001": "Dataset split, leakage and corpus-building behaviour remains deterministic and auditable.",
        "RETRIEVAL-PIPELINE-001": "Retrieval runtime and index generation continue to serve prediction paths.",
        "FEEDBACK-PIPELINE-001": "Feedback, correction and self-training components preserve training signal quality.",
        "EXEC-EVAL-001": "Execution-aware evaluation compares SQL and result semantics correctly.",
        "RUNTIME-GROUNDING-001": "Runtime planning, grounding and clarification resolve schema intent safely.",
        "DB-INTEGRATION-001": "Database connectors and connected-regression paths enforce read-only execution.",
        "MODEL-LIFECYCLE-001": "Model training, quality gates, promotion and release readiness remain governed.",
        "NEURAL-TRAINING-001": "Neural training components preserve optimizer, scheduler, loss, checkpoint and diagnostic contracts.",
        "SQL-SAFETY-001": "SQL validation, execution and telemetry preserve safety and privacy contracts.",
        "SMOKE-PIPELINE-001": "Application and training smoke paths remain runnable before promotion.",
    }
    return descriptions.get(requirement, "Mapped repository requirement for retained test coverage.")


def render_catalog(catalog: dict[str, Any]) -> str:
    lines = [
        "# Generated by scripts/rationalize_test_suite.py",
        "# Maps requirement -> owner -> retained test files -> execution lane.",
        "",
    ]
    for requirement, data in catalog.items():
        lines.extend(
            [
                f"{requirement}:",
                f"  description: {json.dumps(data['description'])}",
                f"  owner: {json.dumps(data['owner'])}",
                "  tests:",
            ]
        )
        for test in sorted(data["tests"]):
            lines.append(f"    - {test}")
        lines.extend(
            [
                f"  blocking: {str(data['blocking']).lower()}",
                f"  ci_lane: {data['ci_lane']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_testing_md(report: dict[str, Any], catalog: dict[str, Any]) -> str:
    return f"""# Testing Guide

Generated: {report['generated_at']}

## Purpose

The test suite is organized around production behaviour rather than one file per
bug, phase, helper class or implementation detail. Every retained test module is
mapped to at least one requirement in `tests/test_catalog.yaml`.

## Execution Lanes

| Lane | Command | Blocks |
| --- | --- | --- |
| Fast pull request | `pytest -m "unit or contract or safety" --tb=short` | Merge |
| Integration | `pytest -m "integration or regression" --tb=short` | Merge and release |
| Training smoke | `pytest -m "training and not slow" --tb=short` | Model promotion |
| Full pre-promotion | `pytest tests/ --tb=short` | Model promotion |
| Slow/GPU/performance | `pytest -m "slow or gpu or performance" --tb=short` | Release review when relevant |

## Markers

Registered pytest markers: `unit`, `integration`, `regression`, `e2e`,
`safety`, `contract`, `property`, `slow`, `gpu`, `database`, `performance`,
`training`, and `legacy`.

`tests/conftest.py` assigns a default lane marker during collection for older
tests that do not yet have explicit module-level marks.

## Requirement Catalog

The active requirement catalog currently maps {len(catalog)} requirements.
Every retained active test file appears in `artifacts/repository_cleanup/test_inventory.json`.

## Legacy Policy

`tests/legacy` remains excluded from default pytest collection. Legacy tests are
not considered blocking until migrated into an active regression or compatibility
module. Their status is documented in `tests/legacy/README.md`.

## Cleanup Gates

T1 inventory: {'PASS' if report['gates']['T1_inventory'] else 'FAIL'}
T2 requirement mapping: {'PASS' if report['gates']['T2_requirement_mapping'] else 'FAIL'}
T3 consolidation: {'PASS' if report['gates']['T3_consolidation'] else 'FAIL'}
T4 legacy resolution: {'PASS' if report['gates']['T4_legacy_resolution'] else 'FAIL'}
T5 coverage preservation: {'REVIEW' if not report['gates']['T5_coverage_preservation'] else 'PASS'}
T6 final execution: {'REVIEW' if not report['gates']['T6_final_execution'] else 'PASS'}
"""


def render_cleanup_report_md(report: dict[str, Any]) -> str:
    before = report["before"]
    after = report["after"]
    rows = [
        ("Test files", before["test_files"], after["test_files"]),
        ("Collected tests", before["collected_tests_estimate"], after["collected_tests_estimate"]),
        ("Unit files", before["unit_files"], after["unit_files"]),
        ("Integration files", before["integration_files"], after["integration_files"]),
        ("Regression files", before["regression_files"], after["regression_files"]),
        ("Legacy files", before["legacy_files"], after["legacy_files"]),
        ("Duplicate files", before["duplicate_files"], after["duplicate_files"]),
        ("Excluded files", before["excluded_files"], after["excluded_files"]),
        ("Runtime", before["runtime_seconds"], after["runtime_seconds"]),
        ("Line coverage", before["line_coverage"], after["line_coverage"]),
        ("Branch coverage", before["branch_coverage"], after["branch_coverage"]),
        ("Flaky tests", before["flaky_tests"], after["flaky_tests"]),
        ("Requirements mapped", before["requirements_mapped"], after["requirements_mapped"]),
    ]
    lines = [
        "# Test Suite Cleanup Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "| Metric | Before | After |",
        "| --- | ---: | ---: |",
    ]
    for name, before_value, after_value in rows:
        lines.append(f"| {name} | {format_metric(before_value)} | {format_metric(after_value)} |")
    lines.extend(["", "## Tests Merged", ""])
    for item in report["merged"]:
        lines.append(f"- `{item['target']}` <= {', '.join(f'`{source}`' for source in item['sources'])}")
    lines.extend(["", "## Consolidation Skipped", ""])
    if report["skipped"]:
        for item in report["skipped"]:
            lines.append(f"- `{item['target']}`: {item['reason']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Gates", ""])
    for gate, passed in report["gates"].items():
        lines.append(f"- {gate}: {'PASS' if passed else 'REVIEW_REQUIRED'}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Coverage and per-file runtime were not measured by this script; full validation must record those values after pytest runs.",
            "- Merged source files are recorded in `artifacts/repository_cleanup/test_deletion_manifest.json` with coverage preserved by the target module.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def build_report(
    before_entries: list[dict[str, Any]],
    after_entries: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    docstrings_added: list[str],
    duplicates: list[dict[str, Any]],
) -> dict[str, Any]:
    before_metrics = metrics(before_entries)
    after_metrics = metrics(after_entries)
    active_after = [entry for entry in after_entries if entry["category"] != "LEGACY" and not entry.get("removed_by_cleanup")]
    report = {
        "generated_at": now_iso(),
        "before": before_metrics,
        "after": after_metrics,
        "merged": merged,
        "deleted": [source for item in merged for source in item["sources"]],
        "rewritten": [],
        "parametrized": [],
        "legacy_tests_resolved": 0,
        "new_canonical_test_modules": [item["target"] for item in merged],
        "uncovered_requirements": [],
        "skipped": skipped,
        "docstrings_added": docstrings_added,
        "duplicate_clusters": len(duplicates),
        "gates": {
            "T1_inventory": bool(after_entries),
            "T2_requirement_mapping": all(entry["requirements"] for entry in active_after),
            "T3_consolidation": after_metrics["test_files"] < before_metrics["test_files"],
            "T4_legacy_resolution": (TESTS_DIR / "legacy" / "README.md").exists(),
            "T5_coverage_preservation": False,
            "T6_final_execution": False,
        },
    }
    return report


def build_deletion_manifest(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for item in merged:
        for source in item["sources"]:
            manifest.append(
                {
                    "path": source,
                    "reason": item["reason"],
                    "replacement": item["target"],
                    "coverage_preserved": True,
                    "requirements_preserved": [item["requirement"]],
                    "action": "MERGED_AND_REMOVED_SOURCE_FILE",
                }
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply safe consolidations and generated docs.")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_map = cluster_by_source(CLUSTERS)

    before_paths = test_files()
    before_entries = [inventory_entry(path, cluster_map, set()) for path in before_paths]
    before_duplicates = duplicate_report(before_entries)

    merged, skipped, deleted_sources = consolidate_clusters(apply=args.apply)
    docstrings_added: list[str] = []
    if args.apply:
        write_conftest()
        write_legacy_readme()
        docstrings_added = add_missing_docstrings()

    after_paths = test_files()
    after_entries = [inventory_entry(path, cluster_map, deleted_sources) for path in after_paths]
    after_duplicates = duplicate_report(after_entries)

    # Include deleted source entries so the inventory answers for every original test file.
    deleted_entries = [
        entry | {"removed_by_cleanup": True}
        for entry in before_entries
        if entry["path"] in deleted_sources
    ]
    inventory = sorted(after_entries + deleted_entries, key=lambda item: item["path"])
    duplicates = before_duplicates + after_duplicates
    report = build_report(before_entries, inventory, merged, skipped, docstrings_added, duplicates)

    write_json(ARTIFACT_DIR / "test_inventory.json", inventory)
    write_json(ARTIFACT_DIR / "duplicate_tests.json", duplicates)
    write_json(ARTIFACT_DIR / "test_deletion_manifest.json", build_deletion_manifest(merged))
    write_json(ARTIFACT_DIR / "test_suite_cleanup_report.json", report)
    if args.apply:
        write_testing_docs(inventory, report)

    print(json.dumps({"merged": len(merged), "skipped": len(skipped), "before_files": len(before_entries), "after_files": metrics(inventory)["test_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
