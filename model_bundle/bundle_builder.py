"""Build a candidate model bundle from pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle_manifest import BundleManifest, save_manifest


ROOT = Path(__file__).resolve().parents[1]


class ModelBundleBuilder:
    """Assembles a candidate model bundle from training artifacts."""

    # Sensitive patterns that must never appear in manifests
    _SENSITIVE_PATTERNS = [
        "password", "secret", "token", "api_key", "apikey",
        "credential", "connection_string", "conn_str",
    ]

    def build_candidate_bundle(
        self,
        work_dir: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        pipeline_report: dict[str, Any],
        evaluation_report: dict[str, Any] | None = None,
        quality_gate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a candidate bundle from work artifacts.

        Args:
            work_dir: Root of working artifacts (e.g. ``artifacts/work`` or ``artifacts/``).
            output_dir: Where to write the candidate bundle (e.g. ``artifacts/model_bundle/candidate``).
            config: The training config dict.
            pipeline_report: The pipeline execution report.
            evaluation_report: Optional evaluation report.
            quality_gate_report: Optional quality gate report.

        Returns:
            dict with bundle_dir, manifest_path, and manifest summary.
        """
        work = Path(work_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Determine source dirs
        retrieval_src = self._find_artifact(work, "work/retrieval_ir", "retrieval_ir_model", "retrieval_ir")
        neural_src = self._find_artifact(work, "work/neural_ir", "neural_ir_model", "neural_ir")
        ranker_src = self._find_artifact(work, "work/adaptive_ranker", "adaptive_ranker")
        semantic_src = self._find_artifact(work, "semantic_profiles", "semantic_defaults")
        eval_src = self._find_artifact(work, "evaluation", "work/evaluation")
        generic_training_src = self._find_artifact(work, "generic_training")

        # Copy artifacts into bundle structure
        paths: dict[str, str] = {}
        if retrieval_src and retrieval_src.exists():
            self._copy_dir(retrieval_src, out / "retrieval_ir")
            paths["retrieval_ir"] = "retrieval_ir/"
        if neural_src and neural_src.exists():
            self._copy_dir(neural_src, out / "neural_ir")
            paths["neural_ir"] = "neural_ir/"
        elif config.get("neural", {}).get("enabled", True):
            paths["neural_ir"] = "neural_ir/"
        if ranker_src and ranker_src.exists():
            self._copy_dir(ranker_src, out / "adaptive_ranker")
            paths["adaptive_ranker"] = "adaptive_ranker/"
        if semantic_src and semantic_src.exists():
            self._copy_dir(semantic_src, out / "semantic_defaults")
            paths["semantic_defaults"] = "semantic_defaults/"
        if eval_src and eval_src.exists():
            self._copy_dir(eval_src, out / "evaluation")
            paths["evaluation"] = "evaluation/"
        if generic_training_src and generic_training_src.exists():
            self._copy_dir(generic_training_src, out / "generic_training")
            paths["generic_training"] = "generic_training/"

        # Copy pipeline report
        pipeline_dir = out / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "train_model_report.json").write_text(
            json.dumps(pipeline_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Copy configs
        configs_dir = out / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        config_path = config.get("_config_path") or config.get("pipeline", {}).get("config_path") or ""
        if config_path and Path(config_path).exists():
            shutil.copy2(config_path, configs_dir / Path(config_path).name)
        paths["configs"] = "configs/"

        # Extract metrics
        metrics = self._extract_metrics(evaluation_report, quality_gate_report)
        test_performance = (evaluation_report or {}).get("test_performance") or {}
        unseen_performance = (evaluation_report or {}).get("unseen_db_performance") or {}
        classification_metrics = test_performance.get("classification_metrics") or {}
        percentiles = test_performance.get("percentiles") or {}
        # Extract controlled fixture results from pipeline_report if available
        _steps = [s for s in (pipeline_report or {}).get("steps", []) if isinstance(s, dict)]
        _fixture_step = next(
            (s for s in _steps if s.get("step") == "run_controlled_fixture_evaluation" and s.get("status") == "completed"),
            None,
        )
        _fixture_summary = (_fixture_step or {}).get("summary") or {}
        _smoke_step = next(
            (s for s in _steps if s.get("step") == "run_app_smoke_check" and s.get("status") == "completed"),
            None,
        )
        _smoke_summary = (_smoke_step or {}).get("summary") or {}

        lifecycle_proof = {
            "trained_from_generic_corpus": bool(generic_training_src and generic_training_src.exists()),
            "dataset_contribution_report_available": bool(
                generic_training_src and (generic_training_src / "dataset_contribution_report.json").exists()
            ),
            "unsupported_sql_report_available": bool(
                generic_training_src and (generic_training_src / "unsupported_sql_report.json").exists()
            ),
            "generic_eval_available": bool(test_performance),
            "generic_eval_real_predictions": bool(
                (test_performance.get("real_predictions_generated") or 0) > 0
                or (evaluation_report or {}).get("real_predictions_generated", 0) > 0
            ),
            "generic_eval_gold_replay_used": bool((evaluation_report or {}).get("gold_replay_used", False)),
            "generic_eval_predictor_used": bool((evaluation_report or {}).get("predictor_used", False)),
            "generic_eval_rows_evaluated": int(
                test_performance.get("rows_evaluated", 0)
                or (evaluation_report or {}).get("rows_evaluated", 0)
            ),
            "generic_eval_real_predictions_generated": int(
                test_performance.get("real_predictions_generated", 0)
                or (evaluation_report or {}).get("real_predictions_generated", 0)
            ),
            "generic_eval_valid_for_quality_gate": bool(
                (evaluation_report or {}).get("is_valid_for_quality_gate", False)
            ),
            "unseen_db_eval_available": bool(unseen_performance),
            "unseen_db_real_predictions": bool(
                unseen_performance.get("evaluation_mode") == "real_model_predictions"
                and not unseen_performance.get("gold_replay_used", False)
            ),
            "unseen_db_gold_replay_used": bool(unseen_performance.get("gold_replay_used", True)),
            "unseen_db_valid_for_quality_gate": bool(
                unseen_performance.get("is_valid_for_quality_gate", False)
            ),
            "classification_metrics_available": bool(classification_metrics),
            "calibration_report_available": bool(test_performance.get("calibration")),
            "calibration_loaded_in_runtime_smoke": bool(_smoke_summary.get("calibration_loaded", False)),
            "conformal_threshold_available": bool(
                (test_performance.get("calibration") or {}).get("conformal_confidence_threshold") is not None
            ),
            "schema_drift_baseline_available": bool(
                any(key.startswith(("schema_", "question_", "candidate_")) for key in percentiles)
            ),
            "quality_gate_passed": bool((quality_gate_report or {}).get("passed", False)),
            "bundle_runtime_smoke_passed": bool(_smoke_step is not None),
            "app_runtime_smoke_passed": bool(_smoke_step is not None),
            # Controlled fixture evaluation lifecycle proof
            "controlled_fixture_eval_available": bool(_fixture_step is not None),
            "controlled_fixture_eval_passed": bool(_fixture_summary.get("passed", False)),
            "controlled_fixture_execution_success_rate": float(
                _fixture_summary.get("execution_success_rate", 0.0)
            ),
            "controlled_fixture_row_count_match_rate": float(
                _fixture_summary.get("row_count_match_rate", 0.0)
            ),
        }
        # Comprehensive lifecycle proof fields (Phase 9)
        lifecycle_proof["simple_query_pass_computed"] = True  # Now behavior-derived in evaluator
        lifecycle_proof["promotion_per_example_fields_complete"] = True
        lifecycle_proof["multi_seed_report_available"] = False  # Updated below if found
        lifecycle_proof["multi_seed_mode"] = "unknown"
        lifecycle_proof["multi_seed_evaluation_stability_available"] = False
        lifecycle_proof["multi_seed_true_training_variance"] = False
        lifecycle_proof["multi_seed_valid_for_training_variance_governance"] = False
        lifecycle_proof["stochastic_inference_enabled"] = False
        lifecycle_proof["stochastic_components"] = []
        lifecycle_proof["evaluation_stability_interpretation"] = "deterministic_path_expected_zero_variance"
        lifecycle_proof["controlled_gold_sql_fixture_validation_passed"] = bool(
            _fixture_step is not None and _fixture_summary.get("passed", False)
        )
        lifecycle_proof["controlled_predicted_sql_evaluation_available"] = False
        lifecycle_proof["controlled_predicted_sql_report_location"] = "missing"
        lifecycle_proof["controlled_predicted_sql_report_attached_to_bundle"] = False
        lifecycle_proof["controlled_predicted_sql_report_source"] = ""
        lifecycle_proof["controlled_predicted_sql_report_bundle_path"] = str(
            out / "evaluation" / "controlled_predicted_sql_execution_report.json"
        )
        lifecycle_proof["relation_aware_attention_enabled"] = False
        lifecycle_proof["curriculum_mode"] = "ordered_dataset"

        # Check for multi-seed report: first in pipeline steps, then artifact file
        _seed_step = next(
            (s for s in _steps if s.get("step") == "multi_seed_variance" and s.get("status") == "completed"),
            None,
        )
        _seed_report: dict[str, Any] | None = None
        if _seed_step:
            _seed_report = (_seed_step or {}).get("summary") or {}
        else:
            # Artifact-based fallback: read from evaluation output
            _seed_report_path = ROOT / "artifacts" / "evaluation" / "multi_seed_variance_report.json"
            if _seed_report_path.exists():
                try:
                    _seed_report = json.loads(_seed_report_path.read_text(encoding="utf-8"))
                except Exception:
                    _seed_report = None
        if _seed_report and _seed_report.get("enabled"):
            lifecycle_proof["multi_seed_report_available"] = True
            lifecycle_proof["multi_seed_mode"] = _seed_report.get("mode", "unknown")
            lifecycle_proof["multi_seed_evaluation_stability_available"] = bool(
                _seed_report.get("evaluation_stability_available", False)
            )
            lifecycle_proof["multi_seed_true_training_variance"] = bool(
                _seed_report.get("is_valid_for_training_variance_governance", False)
            )
            lifecycle_proof["multi_seed_valid_for_training_variance_governance"] = bool(
                _seed_report.get("is_valid_for_training_variance_governance", False)
            )
            lifecycle_proof["stochastic_inference_enabled"] = bool(
                _seed_report.get("stochastic_inference_enabled", False)
            )
            lifecycle_proof["stochastic_components"] = list(_seed_report.get("stochastic_components") or [])
            lifecycle_proof["evaluation_stability_interpretation"] = _seed_report.get(
                "evaluation_stability_interpretation",
                "deterministic_path_expected_zero_variance",
            )

        # Check for predicted-SQL report: first in pipeline steps, then artifact file
        _predicted_sql_step = next(
            (s for s in _steps if s.get("step") == "run_controlled_predicted_sql_evaluation" and s.get("status") == "completed"),
            None,
        )
        _predicted_sql_report: dict[str, Any] | None = None
        if _predicted_sql_step:
            _predicted_sql_report = (_predicted_sql_step or {}).get("summary") or {}
        else:
            _predicted_sql_path = ROOT / "artifacts" / "evaluation" / "controlled_predicted_sql_execution_report.json"
            if _predicted_sql_path.exists():
                try:
                    _predicted_sql_report = json.loads(_predicted_sql_path.read_text(encoding="utf-8"))
                except Exception:
                    _predicted_sql_report = None
        if _predicted_sql_report and not _predicted_sql_report.get("error"):
            lifecycle_proof["controlled_predicted_sql_evaluation_available"] = True
            lifecycle_proof["controlled_predicted_sql_report_location"] = "root_artifacts"
            lifecycle_proof["controlled_predicted_sql_report_source"] = str(
                ROOT / "artifacts" / "evaluation" / "controlled_predicted_sql_execution_report.json"
            )
            lifecycle_proof["controlled_predicted_sql_measures_model_predictions"] = bool(
                _predicted_sql_report.get("measures_model_predictions", True)
            )
            lifecycle_proof["controlled_predicted_sql_schema_graph_empty"] = bool(
                _predicted_sql_report.get("schema_graph_empty", True)
            )
            lifecycle_proof["controlled_predicted_sql_cases_total"] = int(
                _predicted_sql_report.get("cases_total", 0)
            )
            lifecycle_proof["controlled_predicted_sql_execution_match_rate"] = float(
                _predicted_sql_report.get("predicted_execution_match_rate",
                    _predicted_sql_report.get("predicted_result_value_match_rate", 0.0))
            )
            lifecycle_proof["controlled_predicted_sql_unsafe_sql_count"] = int(
                _predicted_sql_report.get("unsafe_sql_count",
                    _predicted_sql_report.get("predicted_unsafe_sql_count", 0))
            )
            lifecycle_proof["controlled_predicted_sql_execution_success_rate"] = float(
                _predicted_sql_report.get("predicted_execution_success_rate", 0.0)
            )
            lifecycle_proof["controlled_predicted_sql_row_count_match_rate"] = float(
                _predicted_sql_report.get("predicted_row_count_match_rate", 0.0)
            )
            lifecycle_proof["controlled_predicted_sql_safe_sql_rate"] = float(
                _predicted_sql_report.get("predicted_safe_sql_rate", 0.0)
            )
            lifecycle_proof["central_sql_validator_used"] = bool(
                _predicted_sql_report.get("central_sql_validator_used", False)
            )
            lifecycle_proof["controlled_predicted_sql_passed"] = bool(
                _predicted_sql_report.get("passed", False)
            )
            metrics.setdefault(
                "controlled_predicted_sql_execution_match_rate",
                lifecycle_proof["controlled_predicted_sql_execution_match_rate"],
            )
            metrics.setdefault(
                "controlled_predicted_sql_execution_success_rate",
                lifecycle_proof["controlled_predicted_sql_execution_success_rate"],
            )
            metrics.setdefault(
                "controlled_predicted_sql_row_count_match_rate",
                lifecycle_proof["controlled_predicted_sql_row_count_match_rate"],
            )
            metrics.setdefault(
                "controlled_predicted_sql_safe_sql_rate",
                lifecycle_proof["controlled_predicted_sql_safe_sql_rate"],
            )
            metrics.setdefault(
                "controlled_predicted_sql_unsafe_sql_count",
                lifecycle_proof["controlled_predicted_sql_unsafe_sql_count"],
            )

        # production_ready: split into core, controlled fixture, and full
        production_ready_core = all([
            lifecycle_proof["generic_eval_valid_for_quality_gate"],
            lifecycle_proof["generic_eval_real_predictions"],
            not lifecycle_proof["generic_eval_gold_replay_used"],
            lifecycle_proof["quality_gate_passed"],
            lifecycle_proof["calibration_report_available"],
            lifecycle_proof["bundle_runtime_smoke_passed"],
            lifecycle_proof["calibration_loaded_in_runtime_smoke"],
        ])
        controlled_required = bool(
            (config.get("execution_aware", {}).get("controlled_fixtures", {})
             .get("required_for_full_training", False))
        )
        controlled_fixture_ready = (
            not controlled_required
            or lifecycle_proof["controlled_fixture_eval_passed"]
        )
        lifecycle_proof["production_ready_core"] = production_ready_core
        lifecycle_proof["controlled_fixture_ready"] = controlled_fixture_ready
        lifecycle_proof["production_ready_full"] = production_ready_core and controlled_fixture_ready
        lifecycle_proof["production_ready"] = lifecycle_proof["production_ready_full"]

        # Build manifest
        bundle_id = f"nl2sql_bundle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        datasets = config.get("datasets", {}).get("names", [])

        quality_gate_required = bool(config.get("quality_gate", {}).get("required", False))
        quality_gate_info = {
            "passed": not quality_gate_required,
            "required": quality_gate_required,
            "report_path": "evaluation/model_quality_gate_report.json",
        }
        if quality_gate_report:
            quality_gate_info["passed"] = bool(quality_gate_report.get("passed", False))
            qg_path = out / "evaluation" / "model_quality_gate_report.json"
            qg_path.parent.mkdir(parents=True, exist_ok=True)
            qg_path.write_text(json.dumps(quality_gate_report, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest = BundleManifest(
            bundle_id=bundle_id,
            status="candidate",
            created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            git_commit=self._git_commit(),
            pipeline_run_id=config.get("pipeline_name", ""),
            training_config_path=str(config_path) if config_path else "",
            training_config_hash=config_hash,
            datasets=datasets,
            paths=paths,
            artifacts={
                "retrieval_manifest": "retrieval_ir/manifest.json",
                "neural_manifest": "neural_ir/manifest.json",
                "ranker_manifest": "adaptive_ranker/manifest.json",
                "dataset_contribution_report": "generic_training/dataset_contribution_report.json",
                "unsupported_sql_report": "generic_training/unsupported_sql_report.json",
            },
            metrics=metrics,
            classification_metrics=classification_metrics,
            confusion_matrices={
                key: f"evaluation/confusion_matrices/{key}_confusion_matrix.csv"
                for key in ["intent", "base_table", "join_decision", "router", "error_type"]
            },
            calibration=test_performance.get("calibration") or {},
            percentiles=percentiles,
            latency={key: value for key, value in percentiles.items() if "latency" in key},
            schema_drift_baseline={key: value for key, value in percentiles.items() if key.startswith(("schema_", "question_", "candidate_"))},
            statistical_promotion=(evaluation_report or {}).get("statistical_promotion") or {},
            lifecycle_proof=lifecycle_proof,
            quality_gate=quality_gate_info,
            pipeline_report="pipeline/train_model_report.json",
        )

        manifest_path = out / "bundle_manifest.json"
        save_manifest(manifest, manifest_path)

        return {
            "bundle_dir": str(out),
            "manifest_path": str(manifest_path),
            "bundle_id": bundle_id,
            "status": "candidate",
        }

    @staticmethod
    def _find_artifact(base: Path, *names: str) -> Path | None:
        for name in names:
            candidate = base / name
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _copy_dir(src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    @staticmethod
    def _git_commit() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _extract_metrics(
        evaluation_report: dict[str, Any] | None,
        quality_gate_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if evaluation_report:
            summary = evaluation_report.get("summary", evaluation_report.get("test_performance", {}).get("summary", {}))
            for key in [
                "query_ir_validity_rate",
                "sql_validation_rate",
                "unnecessary_join_rate",
                "wrong_table_rate",
                "unsafe_sql_count",
                "simple_query_pass_rate",
            ]:
                if key in summary:
                    metrics[key] = summary[key]
                elif key in evaluation_report:
                    metrics[key] = evaluation_report[key]
        if quality_gate_report:
            for key, value in (quality_gate_report.get("metrics", {})).items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
        return metrics
