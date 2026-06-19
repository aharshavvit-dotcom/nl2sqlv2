"""Validate a model bundle against required structure and quality rules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .bundle_manifest import load_manifest


REQUIRED_MANIFEST_METRICS = [
    "query_ir_validity_rate",
    "sql_validation_rate",
    "unsafe_sql_count",
    "unnecessary_join_rate",
    "wrong_table_rate",
]


class ModelBundleValidator:
    """Validates that a model bundle directory is complete and safe."""

    _SENSITIVE_PATTERNS = re.compile(
        r"(password|secret|token|api_key|apikey|credential|connection_string|conn_str)",
        re.IGNORECASE,
    )
    _CREDENTIAL_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s:@]+@", re.IGNORECASE)

    def validate(self, bundle_dir: str | Path) -> dict[str, Any]:
        path = Path(bundle_dir)
        issues: list[str] = []
        warnings: list[str] = []
        checked: list[str] = []

        manifest_path = path / "bundle_manifest.json"
        checked.append(str(manifest_path))
        if not manifest_path.exists():
            issues.append("bundle_manifest.json not found")
            return _result(issues, warnings, checked)

        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            issues.append(f"Failed to parse bundle_manifest.json: {exc}")
            return _result(issues, warnings, checked)

        if manifest.status == "failed":
            issues.append("Bundle status is failed")

        manifest_data = manifest.to_dict()
        _check_no_secrets(manifest_data, issues)

        required_dirs = ["retrieval_ir", "evaluation", "generic_training", "configs"]
        neural_enabled = bool(manifest.paths.get("neural_ir"))
        if neural_enabled:
            required_dirs.append("neural_ir")
        for key in required_dirs:
            rel = manifest.paths.get(key)
            if not rel:
                issues.append(f"Required bundle path missing from manifest: {key}")
                continue
            resolved = path / rel
            checked.append(str(resolved))
            if not resolved.exists():
                issues.append(f"Required artifact folder missing: {key} ({resolved})")

        retrieval_dir = path / manifest.paths.get("retrieval_ir", "retrieval_ir/")
        _require_files(
            retrieval_dir,
            ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"],
            "retrieval",
            issues,
            checked,
        )
        _validate_rag_manifest(retrieval_dir / "manifest.json", manifest.datasets, issues, checked)

        if neural_enabled:
            neural_dir = path / manifest.paths.get("neural_ir", "neural_ir/")
            _require_files(
                neural_dir,
                ["model.pt", "config.yaml", "manifest.json", "vocab.json", "label_maps.json"],
                "neural",
                issues,
                checked,
            )
            _validate_neural_load(neural_dir, issues, warnings)

        eval_dir = path / manifest.paths.get("evaluation", "evaluation/")
        _require_files(eval_dir, ["generic_model_evaluation_report.json"], "evaluation", issues, checked)

        generic_dir = path / manifest.paths.get("generic_training", "generic_training/")
        contribution_path = generic_dir / "dataset_contribution_report.json"
        unsupported_path = generic_dir / "unsupported_sql_report.json"
        _require_files(generic_dir, ["dataset_contribution_report.json", "unsupported_sql_report.json"], "generic_training", issues, checked)
        if contribution_path.exists():
            contribution = _read_json(contribution_path)
            if not contribution.get("leakage_check_passed", False):
                issues.append("Dataset leakage check failed")
            requested = set(contribution.get("datasets_requested") or manifest.datasets or [])
            by_dataset = contribution.get("by_dataset") or {}
            for name in ["spider", "bird-mini"]:
                if name in requested and int((by_dataset.get(name) or {}).get("converted_to_queryir", 0)) <= 0:
                    issues.append(f"Requested dataset contributed zero usable examples: {name}")
        if unsupported_path.exists():
            checked.append(str(unsupported_path))

        qg = manifest.quality_gate or {}
        qg_required = bool(qg.get("required", False))
        qg_path = path / qg.get("report_path", "evaluation/model_quality_gate_report.json")
        checked.append(str(qg_path))
        if qg_required:
            if not qg_path.exists():
                issues.append("Required quality gate report missing")
            elif not _read_json(qg_path).get("passed", False):
                issues.append("Required quality gate failed")
            _require_files(
                eval_dir,
                ["classification_metrics_report.json", "calibration_report.json"],
                "governance evaluation",
                issues,
                checked,
            )
            _require_files(
                eval_dir / "confusion_matrices",
                ["intent_confusion_matrix.csv", "base_table_confusion_matrix.csv", "join_decision_confusion_matrix.csv", "router_confusion_matrix.csv"],
                "confusion matrix",
                issues,
                checked,
            )

        metrics = manifest.metrics or {}
        for key in REQUIRED_MANIFEST_METRICS:
            if key not in metrics:
                issues.append(f"Required manifest metric missing: {key}")
        if float(metrics.get("unsafe_sql_count", 0) or 0) > 0:
            issues.append(f"Unsafe SQL count is {metrics.get('unsafe_sql_count')}, expected 0")
        if float(metrics.get("unnecessary_join_rate", 0.0) or 0.0) > 0.05:
            issues.append(f"Unnecessary join rate is {metrics.get('unnecessary_join_rate')}, max 0.05")
        if float(metrics.get("wrong_table_rate", 0.0) or 0.0) > 0.15:
            issues.append(f"Wrong table rate is {metrics.get('wrong_table_rate')}, max 0.15")
        sql_rate = metrics.get("sql_validation_rate")
        if isinstance(sql_rate, (int, float)) and sql_rate < 0.90:
            issues.append(f"SQL validation rate is {sql_rate}, min 0.90")
        query_ir_rate = metrics.get("query_ir_validity_rate")
        if isinstance(query_ir_rate, (int, float)) and query_ir_rate < 0.90:
            issues.append(f"QueryIR validity rate is {query_ir_rate}, min 0.90")

        _validate_retrieval_runtime(retrieval_dir, neural_dir if neural_enabled else None, issues, warnings)

        return _result(issues, warnings, checked)


def _result(issues: list[str], warnings: list[str], checked: list[str]) -> dict[str, Any]:
    return {
        "passed": len(issues) == 0,
        "blocking_issues": issues,
        "warnings": warnings,
        "checked_files": checked,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _require_files(base: Path, names: list[str], label: str, issues: list[str], checked: list[str]) -> None:
    for name in names:
        target = base / name
        checked.append(str(target))
        if not target.exists():
            issues.append(f"Required {label} artifact missing: {target}")


def _validate_rag_manifest(path: Path, requested_datasets: list[str], issues: list[str], checked: list[str]) -> None:
    checked.append(str(path))
    if not path.exists():
        return
    manifest = _read_json(path)
    for key in ["source_train_file", "total_examples", "by_dataset", "intent_distribution", "sql_complexity_distribution"]:
        if key not in manifest:
            issues.append(f"RAG manifest missing field: {key}")
    by_dataset = manifest.get("by_dataset") or {}
    for name in ["spider", "bird-mini"]:
        if name in set(requested_datasets or []) and int(by_dataset.get(name, 0) or 0) <= 0:
            issues.append(f"RAG manifest shows zero examples for requested dataset: {name}")


def _validate_neural_load(neural_dir: Path, issues: list[str], warnings: list[str]) -> None:
    required = ["model.pt", "config.yaml", "vocab.json", "label_maps.json"]
    if any(not (neural_dir / name).exists() for name in required):
        return
    try:
        from neural_ir.model_registry import load_model_bundle

        load_model_bundle(neural_dir)
    except Exception as exc:
        issues.append(f"Neural model failed load validation: {exc}")


def _validate_retrieval_runtime(
    retrieval_dir: Path,
    neural_dir: Path | None,
    issues: list[str],
    warnings: list[str],
) -> None:
    required = ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"]
    if any(not (retrieval_dir / name).exists() for name in required):
        return
    try:
        from nl2sql_v1.schema import ColumnInfo, SchemaGraph, TableInfo
        from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel

        neural_ready = neural_dir is not None and (neural_dir / "model.pt").exists()
        model = RetrievalNL2SQLModel.load(
            artifact_dir=retrieval_dir,
            neural_ir_model_dir=neural_dir if neural_ready else None,
            allow_dev_fallback=False,
        )
        def table(name: str, columns: dict[str, str]) -> TableInfo:
            return TableInfo(
                name=name,
                columns={
                    column: ColumnInfo(column, typ, True, column == "id")
                    for column, typ in columns.items()
                },
            )

        schema = SchemaGraph(tables={
            "users": table("users", {"id": "integer", "name": "text", "role": "text", "created_at": "timestamp"}),
            "berths": table("berths", {"id": "integer", "berth_name": "text", "berth_code": "text"}),
            "vessels": table("vessels", {"id": "integer", "vessel_name": "text", "vessel_type": "text"}),
            "terminals": table("terminals", {"id": "integer", "terminal_name": "text"}),
            "service_orders": table("service_orders", {"id": "integer", "vessel_id": "integer", "terminal_id": "integer", "status": "text", "cost": "numeric", "created_at": "timestamp"}),
            "assignments": table("assignments", {"id": "integer", "user_id": "integer", "berth_id": "integer", "assigned_date": "date", "status": "text"}),
        }, dialect="sqlite")
        smoke_cases = [
            ("list all users", False),
            ("count users", False),
            ("show users where role is admin", False),
            ("list all berths", False),
            ("show service orders", False),
            ("show assignments with user names", True),
        ]
        for question, join_allowed in smoke_cases:
            result = model.predict(question, schema, use_neural_ir_fallback=False)
            validation = result.validation or {}
            clarified = bool(getattr(result, "needs_clarification", False) or getattr(result, "clarification_questions", []))
            if (validation.get("is_valid") is False or validation.get("ok") is False) and not clarified:
                issues.append(f"Bundle inference smoke returned invalid SQL for {question!r}: {validation}")
            sql = str(result.sql or "")
            if not join_allowed and " join " in f" {sql.lower()} ":
                issues.append(f"Bundle inference smoke added an unnecessary join for {question!r}")
            if sql and not sql.lstrip().lower().startswith(("select", "with")):
                issues.append(f"Bundle inference smoke returned unsafe non-SELECT SQL for {question!r}")
            if not sql and not clarified:
                issues.append(f"Bundle inference smoke produced neither SQL nor clarification for {question!r}")
    except Exception as exc:
        issues.append(f"Bundle runtime smoke failed: {exc}")


def _check_no_secrets(data: Any, issues: list[str]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                if ModelBundleValidator._SENSITIVE_PATTERNS.search(value):
                    issues.append(f"Manifest contains sensitive-looking value at {key}")
                if ModelBundleValidator._CREDENTIAL_URL.search(value):
                    issues.append(f"Manifest contains credential-bearing URL at {key}")
            _check_no_secrets(value, issues)
    elif isinstance(data, list):
        for item in data:
            _check_no_secrets(item, issues)
