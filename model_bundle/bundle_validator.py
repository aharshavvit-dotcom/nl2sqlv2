"""Validate a model bundle against required structure and quality rules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .bundle_manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]

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

    def validate(
        self,
        bundle_dir: str | Path,
        config: dict[str, Any] | None = None,
        *,
        allow_failed_quality_gate_debug: bool = False,
    ) -> dict[str, Any]:
        """Validate bundle structure and production policy.

        ``allow_failed_quality_gate_debug`` downgrades quality/promotion
        failures to warnings for an explicitly requested candidate debug load.
        Artifact integrity, secret checks, model loading, and a manifest status
        of ``failed`` remain blocking.
        """
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

        manifest_data = manifest.to_dict()
        lifecycle_proof = dict(manifest_data.get("lifecycle_proof") or {})

        # Validate report identity consistency (P0 requirement)
        try:
            from orchestration.report_identity import validate_bundle_report_identities
            run_id_val = config.get("_pipeline_run_id") if config else None
            identity_report = validate_bundle_report_identities(path, expected_pipeline_run_id=run_id_val)
            if not identity_report["valid"]:
                if manifest.quality_gate_mode in {"production", "release"}:
                    issues.extend(identity_report["issues"])
                else:
                    warnings.extend(identity_report["issues"])
            for rep_res in identity_report.get("reports") or []:
                rep_name = rep_res["report_name"]
                lifecycle_proof[f"{rep_name}_identity_verified"] = rep_res["valid"]
                lifecycle_proof[f"{rep_name}_pipeline_run_id"] = rep_res["pipeline_run_id"]
                lifecycle_proof[f"{rep_name}_bundle_id"] = rep_res["bundle_id"]
        except Exception as exc:
            warnings.append(f"Failed to run report identity validation: {exc}")

        policy = _controlled_predicted_sql_policy(path, config)
        if allow_failed_quality_gate_debug:
            policy = {
                **policy,
                "required_for_full_training": False,
                "require_report_attached_to_bundle": False,
            }

        if manifest.status == "failed":
            issues.append("Bundle status is failed")

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
            ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json", "sklearn_artifact_metadata.json"],
            "retrieval",
            issues,
            checked,
        )
        _validate_rag_manifest(retrieval_dir / "manifest.json", manifest.datasets, issues, warnings, checked)
        metadata_path = retrieval_dir / "sklearn_artifact_metadata.json"
        if metadata_path.exists():
            try:
                from retrieval.artifact_compatibility import validate_sklearn_metadata

                validate_sklearn_metadata(retrieval_dir, mode="runtime")
            except (OSError, RuntimeError, ValueError) as exc:
                issues.append(str(exc))

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
            diagnostics_path = neural_dir / "training_diagnostics.json"
            if diagnostics_path.exists() and manifest.neural_training_config:
                diagnostics = _read_json(diagnostics_path)
                pairs = {
                    "epochs": "effective_epochs",
                    "batch_size": "effective_batch_size",
                    "save_best_metric": "save_best_metric",
                    "save_best_mode": "save_best_mode",
                    "early_stopping_patience": "early_stopping_patience",
                    "weight_decay": "weight_decay",
                    "pointer_head_weight_decay": "pointer_head_weight_decay",
                    "pointer_dropout": "pointer_dropout",
                    "effective_config_hash": "effective_config_hash",
                }
                mismatches = [
                    key for key, diagnostic_key in pairs.items()
                    if manifest.neural_training_config.get(key) != diagnostics.get(diagnostic_key)
                ]
                if mismatches and manifest.quality_gate_mode in {"production", "release"}:
                    issues.append(
                        "Neural training diagnostics conflict with bundle manifest: "
                        + ", ".join(sorted(mismatches))
                    )

        generic_dir = path / manifest.paths.get("generic_training", "generic_training/")
        contribution_path = generic_dir / "dataset_contribution_report.json"
        unsupported_path = generic_dir / "unsupported_sql_report.json"
        _require_files(generic_dir, ["dataset_contribution_report.json", "unsupported_sql_report.json"], "generic_training", issues, checked)
        if contribution_path.exists():
            contribution = _read_json(contribution_path)
            if not contribution.get("leakage_check_passed", False):
                issues.append("Dataset leakage check failed")
            requested = {str(name).strip().lower() for name in (contribution.get("datasets_requested") or manifest.datasets or []) if str(name).strip()}
            by_dataset = contribution.get("by_dataset") or {}
            for name in ["spider", "bird-mini"]:
                if name in requested and int((by_dataset.get(name) or {}).get("converted_to_queryir", 0)) <= 0:
                    warnings.append(f"Requested dataset contributed zero usable examples: {name}")
            if (
                manifest.quality_gate_mode in {"production", "release"}
                and contribution.get("full_training_dataset_minimums_passed") is not True
            ):
                issues.append("Production dataset contribution minimums failed")
        if unsupported_path.exists():
            checked.append(str(unsupported_path))

        qg = manifest.quality_gate or {}
        qg_required = bool(qg.get("required", False)) and not allow_failed_quality_gate_debug
        qg_path = path / qg.get("report_path", "evaluation/model_quality_gate_report.json")
        checked.append(str(qg_path))

        eval_dir = path / manifest.paths.get("evaluation", "evaluation/")
        _require_files(eval_dir, ["generic_model_evaluation_report.json"], "evaluation", issues, checked)
        evaluation_report_path = eval_dir / "generic_model_evaluation_report.json"
        evaluation_report = _read_json(evaluation_report_path) if evaluation_report_path.exists() else {}
        if evaluation_report:
            _validate_evaluation_source(evaluation_report, issues if qg_required else warnings)
            test_perf = evaluation_report.get("test_performance") or {}
            unseen_perf = evaluation_report.get("unseen_db_performance") or {}
            lifecycle_proof["generic_eval_available"] = True
            lifecycle_proof["generic_eval_real_predictions"] = bool(
                (test_perf.get("real_predictions_generated") or evaluation_report.get("real_predictions_generated") or 0) > 0
            )
            lifecycle_proof["generic_eval_gold_replay_used"] = bool(evaluation_report.get("gold_replay_used", False))
            lifecycle_proof["generic_eval_predictor_used"] = bool(evaluation_report.get("predictor_used", False))
            lifecycle_proof["generic_eval_rows_evaluated"] = int(
                test_perf.get("rows_evaluated", 0) or evaluation_report.get("rows_evaluated", 0)
            )
            lifecycle_proof["generic_eval_real_predictions_generated"] = int(
                test_perf.get("real_predictions_generated", 0) or evaluation_report.get("real_predictions_generated", 0)
            )
            lifecycle_proof["generic_eval_valid_for_quality_gate"] = bool(
                evaluation_report.get("is_valid_for_quality_gate", False)
            )
            lifecycle_proof["unseen_db_eval_available"] = bool(unseen_perf)
            lifecycle_proof["unseen_db_real_predictions"] = bool(
                unseen_perf.get("evaluation_mode") == "real_model_predictions"
                and not unseen_perf.get("gold_replay_used", False)
            )
            lifecycle_proof["unseen_db_gold_replay_used"] = bool(unseen_perf.get("gold_replay_used", True))
            lifecycle_proof["unseen_db_valid_for_quality_gate"] = bool(
                unseen_perf.get("is_valid_for_quality_gate", False)
            )

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
            lifecycle_proof["calibration_report_available"] = (eval_dir / "calibration_report.json").exists()
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
        metric_issues = issues if qg_required else warnings
        if float(metrics.get("unsafe_sql_count", 0) or 0) > 0:
            metric_issues.append(f"Unsafe SQL count is {metrics.get('unsafe_sql_count')}, expected 0")
        if float(metrics.get("unnecessary_join_rate", 0.0) or 0.0) > 0.05:
            metric_issues.append(f"Unnecessary join rate is {metrics.get('unnecessary_join_rate')}, max 0.05")
        if float(metrics.get("wrong_table_rate", 0.0) or 0.0) > 0.15:
            metric_issues.append(f"Wrong table rate is {metrics.get('wrong_table_rate')}, max 0.15")
        sql_rate = metrics.get("sql_validation_rate")
        if isinstance(sql_rate, (int, float)) and sql_rate < 0.90:
            metric_issues.append(f"SQL validation rate is {sql_rate}, min 0.90")
        query_ir_rate = metrics.get("query_ir_validity_rate")
        if isinstance(query_ir_rate, (int, float)) and query_ir_rate < 0.90:
            metric_issues.append(f"QueryIR validity rate is {query_ir_rate}, min 0.90")

        smoke = _validate_retrieval_runtime(retrieval_dir, neural_dir if neural_enabled else None, eval_dir, issues, warnings)
        lifecycle_proof["bundle_runtime_smoke_passed"] = bool(smoke.get("passed", False))
        lifecycle_proof["calibration_loaded_in_runtime_smoke"] = bool(smoke.get("calibration_loaded", False))
        if qg_required:
            if lifecycle_proof.get("generic_eval_gold_replay_used"):
                issues.append("Lifecycle proof shows gold replay was used")
            if not lifecycle_proof.get("generic_eval_real_predictions"):
                issues.append("Lifecycle proof shows no real model predictions were generated")
            if not lifecycle_proof.get("generic_eval_valid_for_quality_gate"):
                issues.append("Lifecycle proof shows generic eval is not valid for quality gate")
            if not lifecycle_proof.get("calibration_report_available"):
                issues.append("Lifecycle proof missing required calibration report")
            if not lifecycle_proof.get("calibration_loaded_in_runtime_smoke"):
                issues.append("Lifecycle proof shows calibration was not loaded in runtime smoke")

        # Check for controlled fixture evaluation results in the bundle
        fixture_report_path = eval_dir / "controlled_fixture_evaluation_report.json"
        if fixture_report_path.exists():
            fixture_report = _read_json(fixture_report_path)
            fixture_summary = fixture_report.get("summary") or {}
            lifecycle_proof["controlled_fixture_eval_available"] = True
            lifecycle_proof["controlled_fixture_eval_passed"] = bool(
                fixture_summary.get("execution_success_rate", 0.0) == 1.0
                and fixture_summary.get("row_count_match_rate", 0.0) == 1.0
                and fixture_summary.get("select_only_rate", 0.0) == 1.0
            )
            lifecycle_proof["controlled_fixture_execution_success_rate"] = float(
                fixture_summary.get("execution_success_rate", 0.0)
            )
            lifecycle_proof["controlled_fixture_row_count_match_rate"] = float(
                fixture_summary.get("row_count_match_rate", 0.0)
            )
            # Validate honest labeling
            lifecycle_proof["controlled_gold_sql_fixture_validation_passed"] = bool(
                lifecycle_proof["controlled_fixture_eval_passed"]
            )
            if fixture_report.get("evaluation_type") != "controlled_gold_sql_fixture_validation":
                warnings.append("Controlled fixture report missing evaluation_type label")
        else:
            lifecycle_proof.setdefault("controlled_fixture_eval_available", False)
            lifecycle_proof.setdefault("controlled_gold_sql_fixture_validation_passed", False)

        # Comprehensive lifecycle proof defaults (Phase 9)
        lifecycle_proof.setdefault("simple_query_pass_computed", True)
        lifecycle_proof.setdefault("promotion_per_example_fields_complete", True)
        lifecycle_proof.setdefault("multi_seed_report_available", False)
        lifecycle_proof.setdefault("multi_seed_mode", "unknown")
        lifecycle_proof.setdefault("multi_seed_evaluation_stability_available", False)
        lifecycle_proof.setdefault("multi_seed_true_training_variance", False)
        lifecycle_proof.setdefault("multi_seed_valid_for_training_variance_governance", False)
        lifecycle_proof.setdefault("controlled_predicted_sql_evaluation_available", False)
        lifecycle_proof.setdefault("controlled_predicted_sql_report_location", "missing")
        lifecycle_proof.setdefault("controlled_predicted_sql_report_attached_to_bundle", False)
        lifecycle_proof.setdefault("controlled_predicted_sql_report_source", "")
        lifecycle_proof.setdefault(
            "controlled_predicted_sql_report_bundle_path",
            str(eval_dir / "controlled_predicted_sql_execution_report.json"),
        )
        lifecycle_proof.setdefault("relation_aware_attention_enabled", False)
        lifecycle_proof.setdefault("curriculum_mode", "ordered_dataset")

        # Read multi-seed variance report from bundle evaluation dir
        seed_report_path = eval_dir / "multi_seed_variance_report.json"
        if seed_report_path.exists():
            seed_report = _read_json(seed_report_path)
            if seed_report.get("enabled"):
                lifecycle_proof["multi_seed_report_available"] = True
                lifecycle_proof["multi_seed_mode"] = seed_report.get("mode", "unknown")
                lifecycle_proof["multi_seed_evaluation_stability_available"] = bool(
                    seed_report.get("evaluation_stability_available", False)
                )
                lifecycle_proof["multi_seed_true_training_variance"] = bool(
                    seed_report.get("is_valid_for_training_variance_governance", False)
                )
                lifecycle_proof["multi_seed_valid_for_training_variance_governance"] = bool(
                    seed_report.get("is_valid_for_training_variance_governance", False)
                )
                lifecycle_proof["stochastic_inference_enabled"] = bool(
                    seed_report.get("stochastic_inference_enabled", False)
                )
                lifecycle_proof["stochastic_components"] = list(seed_report.get("stochastic_components") or [])
                lifecycle_proof["evaluation_stability_interpretation"] = seed_report.get(
                    "evaluation_stability_interpretation",
                    "deterministic_path_expected_zero_variance",
                )
                # Phase 4: Record seed_runs_completed and metric_sample_counts
                lifecycle_proof["seed_runs_completed"] = int(seed_report.get("seed_runs_completed", 0))
                lifecycle_proof["metric_sample_counts"] = dict(seed_report.get("metric_sample_counts") or {})
                # Phase 9: Identity verification for multi-seed report
                seed_identity = _verify_report_identity(
                    seed_report, str(path), manifest, "multi_seed_report",
                )
                lifecycle_proof.update(seed_identity)

        # Read predicted-SQL execution report from bundle first, root artifacts second.
        predicted_sql_report, predicted_location, predicted_path = _read_predicted_sql_report(eval_dir, checked)
        lifecycle_proof["controlled_predicted_sql_report_location"] = predicted_location
        lifecycle_proof["controlled_predicted_sql_report_attached_to_bundle"] = predicted_location == "bundle"
        lifecycle_proof["controlled_predicted_sql_report_bundle_path"] = str(
            eval_dir / "controlled_predicted_sql_execution_report.json"
        )
        lifecycle_proof["controlled_predicted_sql_report_source"] = str(predicted_path) if predicted_path else ""
        if predicted_location == "root_artifacts":
            warnings.append("controlled_predicted_sql_report_not_attached_to_bundle")
            if policy["require_report_attached_to_bundle"]:
                issues.append("controlled_predicted_sql_report_required_but_not_attached_to_bundle")
        elif predicted_location == "missing" and policy["enabled"]:
            warning = "controlled_predicted_sql_report_missing"
            if policy["required_for_full_training"]:
                issues.append(warning)
            else:
                warnings.append(warning)
        if predicted_sql_report:
            # Phase 1: Verify identity metadata
            predicted_identity = _verify_report_identity(
                predicted_sql_report, str(path), manifest,
                "controlled_predicted_sql_report",
            )
            lifecycle_proof.update(predicted_identity)
            if predicted_identity.get("controlled_predicted_sql_report_identity_stale", False):
                warnings.append("controlled_predicted_sql_report_identity_stale")
                if policy["required_for_full_training"]:
                    issues.append("controlled_predicted_sql_report_identity_stale")
            if predicted_identity.get("controlled_predicted_sql_report_identity_missing", False):
                warnings.append("controlled_predicted_sql_report_identity_missing")
                if policy["required_for_full_training"]:
                    issues.append("controlled_predicted_sql_report_identity_missing")
            if predicted_identity.get("controlled_predicted_sql_report_pipeline_run_id_only", False):
                warnings.append("controlled_predicted_sql_report_pipeline_run_id_only")
                if policy["required_for_full_training"]:
                    issues.append("controlled_predicted_sql_report_pipeline_run_id_only")
            if predicted_identity.get("controlled_predicted_sql_report_identity_mismatch", False):
                warnings.append("controlled_predicted_sql_report_identity_mismatch")
                if policy["required_for_full_training"]:
                    issues.append("controlled_predicted_sql_report_identity_mismatch")
            if not predicted_sql_report.get("error"):
                lifecycle_proof["controlled_predicted_sql_evaluation_available"] = True
                lifecycle_proof["controlled_predicted_sql_measures_model_predictions"] = bool(
                    predicted_sql_report.get("measures_model_predictions", True)
                )
                lifecycle_proof["controlled_predicted_sql_schema_graph_empty"] = bool(
                    predicted_sql_report.get("schema_graph_empty", True)
                )
                lifecycle_proof["controlled_predicted_sql_execution_match_rate"] = float(
                    predicted_sql_report.get("predicted_execution_match_rate",
                        predicted_sql_report.get("predicted_result_value_match_rate", 0.0))
                )
                lifecycle_proof["controlled_predicted_sql_unsafe_sql_count"] = int(
                    predicted_sql_report.get("unsafe_sql_count",
                        predicted_sql_report.get("predicted_unsafe_sql_count", 0))
                )
                lifecycle_proof["controlled_predicted_sql_execution_success_rate"] = float(
                    predicted_sql_report.get("predicted_execution_success_rate", 0.0)
                )
                lifecycle_proof["controlled_predicted_sql_row_count_match_rate"] = float(
                    predicted_sql_report.get("predicted_row_count_match_rate", 0.0)
                )
                lifecycle_proof["controlled_predicted_sql_safe_sql_rate"] = float(
                    predicted_sql_report.get("predicted_safe_sql_rate", 0.0)
                )
                lifecycle_proof["central_sql_validator_used"] = bool(
                    predicted_sql_report.get("central_sql_validator_used", False)
                )
                lifecycle_proof["controlled_predicted_sql_passed"] = bool(
                    predicted_sql_report.get("passed", False)
                )
                # Phase 3: failure breakdown
                lifecycle_proof["controlled_predicted_sql_failure_breakdown"] = dict(
                    predicted_sql_report.get("failure_breakdown") or {}
                )
                lifecycle_proof["controlled_predicted_sql_policy_failure_type_counts"] = dict(
                    predicted_sql_report.get("policy_failure_type_counts") or {}
                )
                if predicted_sql_report.get("schema_graph_empty", True):
                    warnings.append("Controlled predicted-SQL evaluation used empty schema graph")
                if policy["required_for_full_training"] and not predicted_sql_report.get("central_sql_validator_used", False):
                    issues.append("controlled_predicted_sql_missing_central_sql_validation")
            elif policy["required_for_full_training"]:
                issues.append(f"controlled_predicted_sql_report_error: {predicted_sql_report.get('error')}")

        # Read execution-aware evaluation report and verify identity (Phase 9)
        exec_aware_path = eval_dir / "execution_aware_evaluation_report.json"
        if exec_aware_path.exists():
            exec_aware_report = _read_json(exec_aware_path)
            exec_identity = _verify_report_identity(
                exec_aware_report, str(path), manifest, "execution_aware_report",
            )
            lifecycle_proof.update(exec_identity)

        # Compute production_ready: split into core/fixture/full
        production_ready_core = bool(
            lifecycle_proof.get("generic_eval_valid_for_quality_gate")
            and lifecycle_proof.get("generic_eval_real_predictions")
            and not lifecycle_proof.get("generic_eval_gold_replay_used")
            and lifecycle_proof.get("quality_gate_passed")
            and lifecycle_proof.get("bundle_runtime_smoke_passed")
            and lifecycle_proof.get("calibration_report_available")
            and lifecycle_proof.get("calibration_loaded_in_runtime_smoke")
            and manifest.quality_gate_mode in {"production", "release"}
            and manifest.quality_gate_passed
            and manifest.eligible_for_promotion
        )
        controlled_fixture_ready = bool(
            lifecycle_proof.get("controlled_fixture_eval_passed", False)
        ) if lifecycle_proof.get("controlled_fixture_eval_available") else True
        lifecycle_proof["production_ready_core"] = production_ready_core
        lifecycle_proof["controlled_fixture_ready"] = controlled_fixture_ready
        lifecycle_proof["production_ready_full"] = production_ready_core and controlled_fixture_ready
        lifecycle_proof["production_ready"] = lifecycle_proof["production_ready_full"]
        lifecycle_proof["quality_gate_mode"] = manifest.quality_gate_mode
        lifecycle_proof["eligible_for_promotion"] = manifest.eligible_for_promotion

        return _result(issues, warnings, checked, lifecycle_proof)


def _result(issues: list[str], warnings: list[str], checked: list[str], lifecycle_proof: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "passed": len(issues) == 0,
        "blocking_issues": issues,
        "warnings": warnings,
        "checked_files": checked,
        "lifecycle_proof": lifecycle_proof or {},
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_predicted_sql_report(eval_dir: Path, checked: list[str]) -> tuple[dict[str, Any] | None, str, Path | None]:
    bundle_path = eval_dir / "controlled_predicted_sql_execution_report.json"
    root_path = ROOT / "artifacts" / "evaluation" / "controlled_predicted_sql_execution_report.json"
    checked.append(str(bundle_path))
    if bundle_path.exists():
        return _read_json(bundle_path), "bundle", bundle_path
    checked.append(str(root_path))
    if root_path.exists():
        return _read_json(root_path), "root_artifacts", root_path
    return None, "missing", None


def _verify_report_identity(
    report: dict[str, Any],
    candidate_bundle_dir: str,
    manifest: Any,  # ModelBundleManifest
    report_name: str,
) -> dict[str, Any]:
    """Phase 1+3+9: Verify report identity metadata matches current candidate."""
    result: dict[str, Any] = {
        f"{report_name}_identity_present": False,
        f"{report_name}_identity_verified": False,
        f"{report_name}_identity_stale": False,
        f"{report_name}_identity_missing": True,
        f"{report_name}_pipeline_run_id_only": False,
        f"{report_name}_identity_mismatch": False,
        f"{report_name}_identity_bundle_id_match": False,
        f"{report_name}_identity_candidate_dir_match": False,
        f"{report_name}_commit_matches": True,
        f"{report_name}_pipeline_run_matches": True,
        f"{report_name}_generated_after_bundle_build": True,
    }
    
    report_bundle_id = report.get("bundle_id")
    report_candidate_dir = report.get("candidate_bundle_dir")
    report_pipeline_run_id = report.get("pipeline_run_id")
    report_commit_sha = report.get("commit_sha")
    report_generated_at = report.get("generated_at")

    strong_identity_present = bool(report_bundle_id) or bool(report_candidate_dir)
    pipeline_run_id_present = bool(report_pipeline_run_id)

    if not strong_identity_present and not pipeline_run_id_present:
        return result

    result[f"{report_name}_identity_present"] = True
    result[f"{report_name}_identity_missing"] = not strong_identity_present

    if not strong_identity_present and pipeline_run_id_present:
        result[f"{report_name}_pipeline_run_id_only"] = True
        result[f"{report_name}_identity_stale"] = True
        return result

    # Check strong identity matches
    bundle_match = False
    if report_bundle_id and manifest.bundle_id:
        bundle_match = str(report_bundle_id) == str(manifest.bundle_id)
    
    dir_match = False
    if report_candidate_dir and candidate_bundle_dir:
        dir_match = str(Path(report_candidate_dir).resolve()) == str(Path(candidate_bundle_dir).resolve())

    identity_verified = bundle_match or dir_match
    
    result[f"{report_name}_identity_bundle_id_match"] = bundle_match
    result[f"{report_name}_identity_candidate_dir_match"] = dir_match
    result[f"{report_name}_identity_verified"] = identity_verified
    result[f"{report_name}_identity_mismatch"] = strong_identity_present and not identity_verified

    # Phase 3 checks
    commit_matches = True
    if report_commit_sha and manifest.git_commit and manifest.git_commit != "unknown":
        commit_matches = str(report_commit_sha) == str(manifest.git_commit)
    result[f"{report_name}_commit_matches"] = commit_matches

    pipeline_run_matches = True
    if report_pipeline_run_id and getattr(manifest, "pipeline_run_id", None):
        pipeline_run_matches = str(report_pipeline_run_id) == str(manifest.pipeline_run_id)
    result[f"{report_name}_pipeline_run_matches"] = pipeline_run_matches

    generated_after = True
    if report_generated_at and manifest.created_at:
        try:
            from dateutil.parser import isoparse
            # Only complain if report was generated BEFORE bundle creation (stale)
            # Some reports are generated after bundle build, which is fine.
            if isoparse(report_generated_at) < isoparse(manifest.created_at):
                generated_after = False
        except Exception:
            pass
    result[f"{report_name}_generated_after_bundle_build"] = generated_after

    stale = not identity_verified or not commit_matches or not pipeline_run_matches or not generated_after
    result[f"{report_name}_identity_stale"] = stale

    return result


def _controlled_predicted_sql_policy(bundle_dir: Path, config: dict[str, Any] | None) -> dict[str, bool]:
    controlled = ((config or {}).get("execution_aware") or {}).get("controlled_predicted_sql") or {}
    if not controlled:
        controlled = _read_bundled_training_config(bundle_dir).get("execution_aware", {}).get("controlled_predicted_sql", {})
    enabled = bool(controlled.get("enabled", False))
    required = bool(controlled.get("required_for_full_training", False))
    return {
        "enabled": enabled,
        "required_for_full_training": required,
        "require_report_attached_to_bundle": bool(
            enabled and required and controlled.get("require_report_attached_to_bundle", True)
        ),
    }


def _read_bundled_training_config(bundle_dir: Path) -> dict[str, Any]:
    configs_dir = bundle_dir / "configs"
    if not configs_dir.exists():
        return {}
    for path in sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(payload, dict) and "execution_aware" in payload:
            return payload
    return {}


def _require_files(base: Path, names: list[str], label: str, issues: list[str], checked: list[str]) -> None:
    for name in names:
        target = base / name
        checked.append(str(target))
        if not target.exists():
            issues.append(f"Required {label} artifact missing: {target}")


def _validate_rag_manifest(path: Path, requested_datasets: list[str], issues: list[str], warnings: list[str], checked: list[str]) -> None:
    checked.append(str(path))
    if not path.exists():
        return
    manifest = _read_json(path)
    for key in ["source_train_file", "total_examples", "by_dataset", "intent_distribution", "sql_complexity_distribution"]:
        if key not in manifest:
            issues.append(f"RAG manifest missing field: {key}")
    by_dataset = manifest.get("by_dataset") or {}
    requested = {str(name).strip().lower() for name in requested_datasets or [] if str(name).strip()}
    for name in ["spider", "bird-mini"]:
        if name in requested and int(by_dataset.get(name, 0) or 0) <= 0:
            warnings.append(f"RAG manifest shows zero examples for requested dataset: {name}")


def _validate_neural_load(neural_dir: Path, issues: list[str], warnings: list[str]) -> None:
    required = ["model.pt", "config.yaml", "vocab.json", "label_maps.json"]
    if any(not (neural_dir / name).exists() for name in required):
        return
    try:
        from neural_ir.model_registry import load_model_bundle

        load_model_bundle(neural_dir)
    except Exception as exc:
        issues.append(f"Neural model failed load validation: {exc}")


def _validate_evaluation_source(report: dict[str, Any], issues: list[str]) -> None:
    sections = [("generic_model_evaluation_report", report)]
    if isinstance(report.get("test_performance"), dict):
        sections.append(("test_performance", report["test_performance"]))
    if isinstance(report.get("unseen_db_performance"), dict):
        sections.append(("unseen_db_performance", report["unseen_db_performance"]))
    for name, section in sections:
        mode = section.get("evaluation_mode")
        if section.get("gold_replay_used") or section.get("gold_replay_baseline"):
            issues.append(f"{name} was generated from gold replay and is not valid for bundle validation")
        if section.get("is_valid_for_quality_gate") is False:
            issues.append(f"{name} is marked not valid for quality gate")
        if mode in {"explicit_gold_replay_baseline", "explicit_oracle_upper_bound"}:
            issues.append(f"{name} uses non-production evaluation mode: {mode}")
        artifact_source = section.get("model_artifact_source")
        if artifact_source == "neural_only_artifact_dirs":
            issues.append(
                f"{name} used neural-only artifact dirs, not the full bundle runtime. "
                "This is acceptable for diagnostics but not for production bundle validation."
            )


def _validate_retrieval_runtime(
    retrieval_dir: Path,
    neural_dir: Path | None,
    eval_dir: Path,
    issues: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    summary = {"passed": False, "calibration_loaded": False}
    issue_count_before = len(issues)
    required = ["example_index.pkl", "schema_index.pkl", "pattern_index.pkl", "manifest.json"]
    if any(not (retrieval_dir / name).exists() for name in required):
        return summary
    try:
        from nl2sql_v1.schema import ColumnInfo, SchemaGraph, TableInfo
        from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel

        neural_ready = neural_dir is not None and (neural_dir / "model.pt").exists()
        calibration_path = eval_dir / "calibration_report.json"
        model = RetrievalNL2SQLModel.load(
            artifact_dir=retrieval_dir,
            neural_ir_model_dir=neural_dir if neural_ready else None,
            confidence_calibration_path=calibration_path if calibration_path.exists() else None,
            allow_dev_fallback=False,
        )
        summary["calibration_loaded"] = bool(model.orchestrator.confidence_calibration)
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
        if neural_ready:
            result = model.predict("list all users", schema, use_neural_ir_fallback=True)
            validation = result.validation or {}
            clarified = bool(getattr(result, "needs_clarification", False) or getattr(result, "clarification_questions", []))
            if (validation.get("is_valid") is False or validation.get("ok") is False) and not clarified:
                issues.append(f"Neural-enabled bundle smoke returned invalid SQL: {validation}")
        summary["passed"] = len(issues) == issue_count_before
    except Exception as exc:
        issues.append(f"Bundle runtime smoke failed: {exc}")
    return summary


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
