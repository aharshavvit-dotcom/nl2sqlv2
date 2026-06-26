from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .pipeline_config import PipelineConfig
from .step_contract import StepContract


ROOT = Path(__file__).resolve().parents[1]

STEP_ALIASES = {
    "train_neural_ir_model": "train_neural_ir",
    "train_ranking_from_gold": "train_adaptive_ranker",
    "run_model_quality_gate": "run_quality_gate",
}


class StepRunner:
    def get_contract(self, step: str, config: PipelineConfig) -> StepContract:
        """Return the contract for a pipeline step."""
        canonical = _canonical_step(step)
        factory = getattr(self, f"_contract_{canonical}", None)
        if factory is not None:
            return factory(config)
        if getattr(self, f"_run_{canonical}", None) is not None:
            return StepContract(name=step, required=True)
        raise ValueError(f"Unknown pipeline step: {step}")

    def run_step(self, step: str, config: PipelineConfig) -> dict[str, Any]:
        canonical = _canonical_step(step)
        if canonical in {"build_generic_ir_corpus", "build_retrieval_rag_index", "train_neural_ir"} and config.skip_heavy_steps:
            return {"status": "skipped", "reason": "skip_heavy_steps enabled for smoke pipeline"}
        method = getattr(self, f"_run_{canonical}", None)
        if method is None:
            raise ValueError(f"Unknown pipeline step: {step}")
        return method(config)

    # Contracts

    def _contract_verify_datasets(self, config: PipelineConfig) -> StepContract:
        return StepContract(name="verify_datasets", required=True)

    def _contract_build_generic_ir_corpus(self, config: PipelineConfig) -> StepContract:
        return StepContract(
            name="build_generic_ir_corpus",
            required=True,
            outputs=[
                str(ROOT / "data/processed/generic_ir_train.jsonl"),
                str(ROOT / "data/processed/generic_ir_validation.jsonl"),
                str(ROOT / "data/processed/generic_ir_test.jsonl"),
                str(ROOT / "data/processed/generic_ir_unseen_db_test.jsonl"),
                str(ROOT / "data/processed/generic_ir_unsupported.jsonl"),
                str(ROOT / "artifacts/generic_training/dataset_contribution_report.json"),
                str(ROOT / "artifacts/generic_training/unsupported_sql_report.json"),
            ],
        )

    def _contract_build_retrieval_rag_index(self, config: PipelineConfig) -> StepContract:
        artifacts = _artifacts(config)
        output = Path(artifacts["retrieval_model_dir"])
        return StepContract(
            name="build_retrieval_rag_index",
            required=True,
            inputs=[str(ROOT / "data/processed/generic_ir_train.jsonl")],
            outputs=[
                str(output / "example_index.pkl"),
                str(output / "schema_index.pkl"),
                str(output / "pattern_index.pkl"),
                str(output / "manifest.json"),
            ],
        )

    def _contract_build_hard_negative_corpus(self, config: PipelineConfig) -> StepContract:
        return StepContract(
            name="build_hard_negative_corpus",
            required=True,
            inputs=[str(ROOT / "data/processed/generic_ir_train.jsonl")],
            outputs=[str(ROOT / "data/processed/generic_ir_hard_negatives.jsonl")],
        )

    def _contract_train_neural_ir(self, config: PipelineConfig) -> StepContract:
        artifacts = _artifacts(config)
        output = Path(artifacts["neural_model_dir"])
        return StepContract(
            name="train_neural_ir",
            required=True,
            inputs=[
                str(ROOT / "data/processed/generic_ir_train.jsonl"),
                str(ROOT / "data/processed/generic_ir_validation.jsonl"),
            ],
            outputs=[
                str(output / "model.pt"),
                str(output / "training_metrics.json"),
            ],
        )

    def _contract_evaluate_against_gold(self, config: PipelineConfig) -> StepContract:
        artifacts = _artifacts(config)
        return StepContract(
            name="evaluate_against_gold",
            required=True,
            inputs=[str(ROOT / "data/processed/generic_ir_validation.jsonl")],
            outputs=[
                str(Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl"),
                str(Path(artifacts["self_training_dir"]) / "validation_gold_comparison_report.json"),
            ],
        )

    def _contract_mine_validation_errors(self, config: PipelineConfig) -> StepContract:
        if not (config.training.get("_integrated_config") or {}).get("self_training", {}).get("enabled", False):
            return StepContract(name="mine_validation_errors", required=False, can_skip=True, skip_reason="self_training disabled in config")
        return StepContract(
            name="mine_validation_errors",
            required=True,
            inputs=[str(Path(_artifacts(config)["self_training_dir"]) / "validation_predictions.jsonl")],
            outputs=[str(ROOT / "data/processed/self_training/error_summary.json")],
        )

    def _contract_build_corrections_from_gold(self, config: PipelineConfig) -> StepContract:
        if not (config.training.get("_integrated_config") or {}).get("self_training", {}).get("enabled", False):
            return StepContract(name="build_corrections_from_gold", required=False, can_skip=True, skip_reason="self_training disabled in config")
        return StepContract(
            name="build_corrections_from_gold",
            required=True,
            inputs=[str(Path(_artifacts(config)["self_training_dir"]) / "validation_predictions.jsonl")],
            outputs=[str(ROOT / "data/processed/self_training/correction_summary.json")],
        )

    def _contract_train_adaptive_ranker(self, config: PipelineConfig) -> StepContract:
        enabled = (config.training.get("_integrated_config") or {}).get("ranker", {}).get("enabled", False)
        if not enabled:
            return StepContract(name="train_adaptive_ranker", required=False, can_skip=True, skip_reason="ranker disabled in config")
        output = Path(_artifacts(config).get("adaptive_ranker_dir", ROOT / "artifacts/work/adaptive_ranker"))
        return StepContract(
            name="train_adaptive_ranker",
            required=True,
            inputs=[str(Path(_artifacts(config)["self_training_dir"]) / "validation_predictions.jsonl")],
            outputs=[str(output / "manifest.json")],
        )

    def _contract_train_ranking_from_gold(self, config: PipelineConfig) -> StepContract:
        return self._contract_train_adaptive_ranker(config)

    def _contract_run_self_improvement_loop(self, config: PipelineConfig) -> StepContract:
        enabled = (config.training.get("_integrated_config") or {}).get("self_training", {}).get("enabled", False)
        if not enabled:
            return StepContract(name="run_self_improvement_loop", required=False, can_skip=True, skip_reason="self_training disabled in config")
        return StepContract(name="run_self_improvement_loop", required=False)

    def _contract_run_execution_aware_evaluation(self, config: PipelineConfig) -> StepContract:
        return StepContract(name="run_execution_aware_evaluation", required=False)

    def _contract_run_controlled_fixture_evaluation(self, config: PipelineConfig) -> StepContract:
        integrated = config.training.get("_integrated_config") or {}
        controlled = (integrated.get("execution_aware") or {}).get("controlled_fixtures") or {}
        if not controlled.get("enabled", False):
            return StepContract(
                name="run_controlled_fixture_evaluation", required=False,
                can_skip=True, skip_reason="controlled_fixtures disabled in config",
            )
        output = controlled.get("output", "artifacts/evaluation/controlled_fixture_evaluation_report.json")
        return StepContract(
            name="run_controlled_fixture_evaluation",
            required=False,
            outputs=[str(ROOT / output)],
        )

    def _contract_evaluate_generic_models(self, config: PipelineConfig) -> StepContract:
        return StepContract(
            name="evaluate_generic_models",
            required=True,
            inputs=[
                str(ROOT / "data/processed/generic_ir_test.jsonl"),
                str(ROOT / "data/processed/generic_ir_unseen_db_test.jsonl"),
            ],
            outputs=[str(Path(_artifacts(config)["evaluation_dir"]) / "generic_model_evaluation_report.json")],
        )

    def _contract_run_quality_gate(self, config: PipelineConfig) -> StepContract:
        required = (config.training.get("_integrated_config") or {}).get("quality_gate", {}).get("required", False)
        if not required:
            return StepContract(name="run_quality_gate", required=False, can_skip=True, skip_reason="quality_gate.required is false")
        return StepContract(
            name="run_quality_gate",
            required=True,
            inputs=[str(Path(_artifacts(config)["evaluation_dir"]) / "generic_model_evaluation_report.json")],
            outputs=[str(Path(_artifacts(config)["evaluation_dir"]) / "model_quality_gate_report.json")],
        )

    def _contract_build_model_bundle(self, config: PipelineConfig) -> StepContract:
        candidate = _candidate_bundle_dir(config)
        return StepContract(
            name="build_model_bundle",
            required=True,
            outputs=[str(candidate / "bundle_manifest.json")],
        )

    def _contract_validate_model_bundle(self, config: PipelineConfig) -> StepContract:
        candidate = _candidate_bundle_dir(config)
        return StepContract(
            name="validate_model_bundle",
            required=True,
            inputs=[str(candidate / "bundle_manifest.json")],
            outputs=[str(candidate / "bundle_validation_report.json")],
        )

    def _contract_attach_runtime_evaluation_reports_to_bundle(self, config: PipelineConfig) -> StepContract:
        integrated = config.training.get("_integrated_config") or {}
        controlled_predicted = (integrated.get("execution_aware") or {}).get("controlled_predicted_sql") or {}
        required = bool(
            controlled_predicted.get("enabled", False)
            and controlled_predicted.get("required_for_full_training", False)
            and controlled_predicted.get("require_report_attached_to_bundle", True)
        )
        candidate = _candidate_bundle_dir(config)
        return StepContract(
            name="attach_runtime_evaluation_reports_to_bundle",
            required=required,
            inputs=[str(candidate / "bundle_manifest.json")],
            outputs=[str(candidate / "evaluation" / "controlled_predicted_sql_execution_report.json")],
        )

    def _contract_promote_model_bundle(self, config: PipelineConfig) -> StepContract:
        should_promote = (config.training.get("_integrated_config") or {}).get("bundle", {}).get("promote_if_quality_gate_passes", False)
        if not should_promote:
            return StepContract(name="promote_model_bundle", required=False, can_skip=True, skip_reason="promotion disabled in config")
        return StepContract(name="promote_model_bundle", required=False)

    # Runners

    def _run_verify_datasets(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.dataset_registry import DatasetRegistry

        requested = [str(item) for item in config.datasets.get("names", [])]
        registry_report = DatasetRegistry().validate_dataset_presence(requested)
        missing = [name for name, report in registry_report.items() if not report.get("available")]
        allow_missing = bool((config.training.get("_integrated_config") or {}).get("datasets", {}).get("allow_missing_dataset", False))
        if missing and not allow_missing:
            raise FileNotFoundError(
                "Required datasets are missing: "
                + ", ".join(missing)
                + ". Run python scripts/download_datasets.py --datasets "
                + " ".join(missing)
            )
        return {"status": "completed", "summary": {"requested": requested, "missing": missing, "allow_missing_dataset": allow_missing}}

    def _run_build_generic_ir_corpus(self, config: PipelineConfig) -> dict[str, Any]:
        from training.build_generic_ir_corpus import build_generic_ir_corpus

        integrated = config.training.get("_integrated_config") or {}
        dataset_cfg = integrated.get("datasets", {})
        args = argparse.Namespace(
            datasets=",".join(config.datasets.get("names", [])),
            max_examples=config.datasets.get("max_examples"),
            max_examples_per_dataset=dataset_cfg.get("max_examples_per_dataset") or config.datasets.get("max_examples_per_dataset"),
            min_converted_examples_required=(
                dataset_cfg.get("min_converted_examples_required")
                or config.datasets.get("min_converted_examples_required")
            ),
            output_dir=ROOT / "data" / "processed",
            artifact_dir=ROOT / "artifacts" / "generic_training",
            seed=config.seed,
            train_ratio=float(dataset_cfg.get("train_ratio", 0.8)),
            validation_ratio=float(dataset_cfg.get("validation_ratio", 0.1)),
            test_ratio=float(dataset_cfg.get("test_ratio", 0.1)),
            unseen_db_test_ratio=float(dataset_cfg.get("unseen_db_test_ratio", 0.15)),
            include_unsupported=True,
            schema_renaming=(integrated.get("augmentation") or {}).get("schema_renaming") or {},
        )
        report = build_generic_ir_corpus(args)
        _enforce_dataset_contribution(report, config)
        contribution = report.get("dataset_contribution_report") or {}
        return {
            "status": "completed",
            "summary": {
                "train_examples": contribution.get("total_training_examples", 0),
                "datasets": config.datasets.get("names", []),
            },
        }

    def _run_build_retrieval_rag_index(self, config: PipelineConfig) -> dict[str, Any]:
        from retrieval.rag_index_builder import RAGIndexBuilder

        report = RAGIndexBuilder().build_from_jsonl(
            ROOT / "data/processed/generic_ir_train.jsonl",
            Path(_artifacts(config)["retrieval_model_dir"]),
        )
        return {"status": "completed", "summary": report}

    def _run_build_hard_negative_corpus(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.hard_negative_corpus_builder import HardNegativeCorpusBuilder
        from dataset_training.utils import read_jsonl, write_jsonl

        input_path = ROOT / "data/processed/generic_ir_train.jsonl"
        output_path = ROOT / "data/processed/generic_ir_hard_negatives.jsonl"
        examples = read_jsonl(input_path)
        negatives = HardNegativeCorpusBuilder().build(examples)
        write_jsonl(output_path, negatives)
        return {
            "status": "completed",
            "summary": {
                "input": str(input_path),
                "output": str(output_path),
                "training_examples": len(examples),
                "hard_negative_examples": len(negatives),
            },
        }

    def _run_train_neural_ir(self, config: PipelineConfig) -> dict[str, Any]:
        from neural_optimization.training_config import load_training_config, merge_cli_overrides
        from training.train_neural_ir_optimized import run_optimized_training

        artifacts = _artifacts(config)
        neural_config = ROOT / str(config.training.get("neural_config") or "configs/neural_training_default.yaml")
        training_config = load_training_config(neural_config)
        training_config = merge_cli_overrides(
            training_config,
            {
                "train": str(ROOT / "data/processed/generic_ir_train.jsonl"),
                "validation": str(ROOT / "data/processed/generic_ir_validation.jsonl"),
                "output_dir": artifacts["neural_model_dir"],
                "epochs": config.training.get("neural_epochs"),
                "batch_size": config.training.get("batch_size"),
                "max_examples": config.datasets.get("max_examples"),
                "hard_negatives": str(ROOT / "data/processed/generic_ir_hard_negatives.jsonl"),
            },
        )
        report = run_optimized_training(training_config, Path(artifacts["neural_model_dir"]))
        if report.get("error"):
            return {"status": "failed", "error": str(report["error"])}
        return {"status": "completed", "summary": report}

    def _run_audit_execution_pipeline(self, config: PipelineConfig) -> dict[str, Any]:
        from scripts.audit_execution_pipeline_readiness import run_audit

        report = run_audit()
        return {"status": "completed", "overall_status": report.get("overall_status")}

    def _run_audit_self_training(self, config: PipelineConfig) -> dict[str, Any]:
        from scripts.audit_self_training_readiness import run_audit

        report = run_audit()
        return {"status": "completed", "overall_status": report.get("overall_status")}

    def _run_evaluate_against_gold(self, config: PipelineConfig) -> dict[str, Any]:
        from training.evaluate_against_gold import _Args, evaluate_against_gold

        artifacts = _artifacts(config)
        args = _Args(
            input=ROOT / "data/processed/generic_ir_validation.jsonl",
            retrieval_model_dir=Path(artifacts["retrieval_model_dir"]),
            neural_model_dir=Path(artifacts["neural_model_dir"]),
            output=Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl",
            report=Path(artifacts["self_training_dir"]) / "validation_gold_comparison_report.json",
            max_examples=int(config.training.get("max_self_training_examples") or config.datasets.get("max_examples") or 100),
        )
        report = evaluate_against_gold(args)
        return {"status": "completed", "summary": report["summary"]}

    def _run_mine_validation_errors(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import read_jsonl, write_json, write_jsonl
        from self_training.hard_negative_miner import HardNegativeMiner

        predictions = Path(_artifacts(config)["self_training_dir"]) / "validation_predictions.jsonl"
        result = HardNegativeMiner().mine(read_jsonl(predictions))
        output = ROOT / "data/processed/self_training"
        write_jsonl(output / "mined_hard_negatives.jsonl", result["mined_hard_negatives"])
        write_json(output / "error_summary.json", result["error_summary"])
        return {"status": "completed", "summary": result["error_summary"]}

    def _run_build_corrections_from_gold(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import read_jsonl, write_json, write_jsonl
        from self_training.correction_builder import CorrectionBuilder

        predictions = Path(_artifacts(config)["self_training_dir"]) / "validation_predictions.jsonl"
        result = CorrectionBuilder().build(read_jsonl(predictions))
        output = ROOT / "data/processed/self_training"
        write_jsonl(output / "correction_positive_examples.jsonl", result["correction_positive_examples"])
        write_jsonl(output / "queryir_repair_examples.jsonl", result["queryir_repair_examples"])
        write_json(output / "correction_summary.json", result["summary"])
        return {"status": "completed", "summary": result["summary"]}

    def _run_train_adaptive_ranker(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import read_jsonl
        from self_training.ranking_trainer import RankingTrainer

        artifacts = _artifacts(config)
        output_dir = Path(artifacts.get("adaptive_ranker_dir", ROOT / "artifacts/work/adaptive_ranker"))
        report = RankingTrainer().train(read_jsonl(Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl"), output_dir)
        manifest = {
            "created_from": str(Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl"),
            "training_rows": report.get("training_rows", 0),
            "feature_names": report.get("feature_names", []),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"status": "completed", "summary": report}

    def _run_train_ranking_from_gold(self, config: PipelineConfig) -> dict[str, Any]:
        return self._run_train_adaptive_ranker(config)

    def _run_run_self_improvement_loop(self, config: PipelineConfig) -> dict[str, Any]:
        from self_training.self_improvement_loop import SelfImprovementLoop

        artifacts = _artifacts(config)
        report = SelfImprovementLoop().run(
            train_path=ROOT / "data/processed/generic_ir_train.jsonl",
            validation_path=ROOT / "data/processed/generic_ir_validation.jsonl",
            retrieval_model_dir=artifacts["retrieval_model_dir"],
            neural_model_dir=artifacts["neural_model_dir"],
            output_dir=artifacts["self_training_dir"],
            iterations=int(config.training.get("self_improvement_iterations", 1)),
            max_examples=int(config.training.get("max_self_training_examples") or config.datasets.get("max_examples") or 100),
        )
        return {"status": "completed", "summary": {"iterations": report["iterations"], "improved": report["improved"]}}

    def _run_run_execution_aware_evaluation(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import read_jsonl, write_json
        from training.run_execution_aware_evaluation import evaluate_rows

        artifacts = _artifacts(config)
        rows = read_jsonl(Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl")
        report = evaluate_rows(rows)
        output = Path(artifacts["evaluation_dir"]) / "execution_aware_evaluation_report.json"
        write_json(output, report)
        return {"status": "completed", "summary": report["summary"]}

    def _run_run_controlled_fixture_evaluation(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import write_json
        from training.run_execution_aware_evaluation import evaluate_controlled_fixtures

        integrated = config.training.get("_integrated_config") or {}
        controlled = (integrated.get("execution_aware") or {}).get("controlled_fixtures") or {}
        output_path = controlled.get("output", "artifacts/evaluation/controlled_fixture_evaluation_report.json")
        report = evaluate_controlled_fixtures()
        output = ROOT / output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        summary = report.get("summary") or {}
        passed = (
            summary.get("execution_success_rate", 0.0) == 1.0
            and summary.get("row_count_match_rate", 0.0) == 1.0
            and summary.get("select_only_rate", 0.0) == 1.0
        )
        return {
            "status": "completed",
            "summary": {
                "controlled_fixture_eval": True,
                "total_cases": report.get("total_cases", 0),
                "execution_success_rate": summary.get("execution_success_rate", 0.0),
                "row_count_match_rate": summary.get("row_count_match_rate", 0.0),
                "safe_sql_rate": summary.get("select_only_rate", 0.0),
                "passed": passed,
            },
        }

    def _contract_run_controlled_predicted_sql_evaluation(self, config: PipelineConfig) -> StepContract:
        integrated = config.training.get("_integrated_config") or {}
        controlled_predicted = (integrated.get("execution_aware") or {}).get("controlled_predicted_sql") or {}
        output = Path(_artifacts(config)["evaluation_dir"]) / "controlled_predicted_sql_execution_report.json"
        if controlled_predicted.get("enabled", False):
            return StepContract(
                name="run_controlled_predicted_sql_evaluation", required=False,
                outputs=[str(output)],
            )
        return StepContract(
            name="run_controlled_predicted_sql_evaluation",
            required=False,
        )

    def _run_run_controlled_predicted_sql_evaluation(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import write_json
        from training.run_execution_aware_evaluation import evaluate_controlled_predicted_sql

        integrated = config.training.get("_integrated_config") or {}
        artifacts = _artifacts(config)
        controlled_predicted = (integrated.get("execution_aware") or {}).get("controlled_predicted_sql") or {}
        bundle_dir = _candidate_bundle_dir(config)

        report = evaluate_controlled_predicted_sql(
            model_artifact_dir=bundle_dir,
            config=controlled_predicted,
        )
        output = Path(artifacts["evaluation_dir"]) / "controlled_predicted_sql_execution_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)

        if report.get("error"):
            return {
                "status": "completed",
                "summary": {
                    "controlled_predicted_sql_eval": True,
                    "error": report["error"],
                    "measures_model_predictions": True,
                    "passed": False,
                },
            }

        return {
            "status": "completed",
            "summary": {
                "controlled_predicted_sql_eval": True,
                "report_path": str(output),
                "measures_model_predictions": True,
                "cases_total": report.get("cases_total", 0),
                "predictions_generated": report.get("predictions_generated", 0),
                "predicted_execution_match_rate": report.get("predicted_execution_match_rate", 0.0),
                "predicted_row_count_match_rate": report.get("predicted_row_count_match_rate", 0.0),
                "predicted_safe_sql_rate": report.get("predicted_safe_sql_rate", 0.0),
                "unsafe_sql_count": report.get("unsafe_sql_count", 0),
                "passed": report.get("passed", False),
            },
        }

    def _run_attach_runtime_evaluation_reports_to_bundle(self, config: PipelineConfig) -> dict[str, Any]:
        import shutil

        integrated = config.training.get("_integrated_config") or {}
        controlled_predicted = (integrated.get("execution_aware") or {}).get("controlled_predicted_sql") or {}
        artifacts = _artifacts(config)
        candidate = _candidate_bundle_dir(config)
        evaluation_dir = Path(artifacts["evaluation_dir"])
        bundle_eval_dir = candidate / "evaluation"
        bundle_eval_dir.mkdir(parents=True, exist_ok=True)

        report_names = [
            "controlled_predicted_sql_execution_report.json",
            "controlled_fixture_evaluation_report.json",
            "multi_seed_variance_report.json",
            "execution_aware_evaluation_report.json",
        ]
        attached: list[dict[str, str]] = []
        missing: list[str] = []
        for name in report_names:
            source = evaluation_dir / name
            if not source.exists():
                missing.append(name)
                continue
            target = bundle_eval_dir / name
            shutil.copy2(source, target)
            attached.append({"name": name, "source": str(source), "bundle_path": str(target)})

        predicted_attached = any(item["name"] == "controlled_predicted_sql_execution_report.json" for item in attached)
        required = bool(
            controlled_predicted.get("enabled", False)
            and controlled_predicted.get("required_for_full_training", False)
            and controlled_predicted.get("require_report_attached_to_bundle", True)
        )
        summary = {
            "attached_reports": attached,
            "missing_optional_reports": missing,
            "controlled_predicted_sql_report_attached_to_bundle": predicted_attached,
            "controlled_predicted_sql_report_source": str(evaluation_dir / "controlled_predicted_sql_execution_report.json"),
            "controlled_predicted_sql_report_bundle_path": str(bundle_eval_dir / "controlled_predicted_sql_execution_report.json"),
        }
        if required and not predicted_attached:
            return {
                "status": "failed",
                "error": "controlled_predicted_sql_report_required_but_missing",
                "summary": summary,
            }
        if controlled_predicted.get("enabled", False) and not predicted_attached:
            summary["warning"] = "controlled_predicted_sql_report_missing_optional"
        return {"status": "completed", "summary": summary}

    def _run_evaluate_generic_models(self, config: PipelineConfig) -> dict[str, Any]:
        from training.evaluate_generic_models import evaluate_generic_models

        artifacts = _artifacts(config)
        integrated = config.training.get("_integrated_config") or {}
        calibration = integrated.get("calibration") or {}
        args = argparse.Namespace(
            test=ROOT / "data/processed/generic_ir_test.jsonl",
            unseen_db_test=ROOT / "data/processed/generic_ir_unseen_db_test.jsonl",
            model_bundle_dir=Path(artifacts["bundle_dir"]) if artifacts.get("bundle_dir") else None,
            retrieval_model_dir=Path(artifacts["retrieval_model_dir"]),
            neural_model_dir=Path(artifacts["neural_model_dir"]),
            output=Path(artifacts["evaluation_dir"]) / "generic_model_evaluation_report.json",
            thresholds=ROOT / "evaluation/model_quality_thresholds.yaml",
            allow_gold_replay_baseline=config.smoke,
            max_examples=config.datasets.get("max_examples") if config.smoke else None,
            calibration_coverage_target=float(calibration.get("abstention_coverage_target", 0.95)),
            use_conformal_threshold=bool(calibration.get("use_conformal_threshold", True)),
            abstain_when_calibrated_confidence_below=calibration.get("abstain_when_calibrated_confidence_below"),
        )
        report = evaluate_generic_models(args)
        return {"status": "completed", "summary": report.get("summary", {})}

    def _run_select_best_model(self, config: PipelineConfig) -> dict[str, Any]:
        from datetime import datetime, timezone

        from model_selection.model_candidate import ModelCandidate
        from model_selection.model_selector import ModelSelector
        from model_selection.selection_reporter import SelectionReporter
        from quality_gates.thresholds import load_thresholds
        from training.select_best_model import _attach_predicted_sql_metrics, _metrics, _read

        artifacts = _artifacts(config)
        controlled_predicted_sql_report = _read(Path(artifacts["evaluation_dir"]) / "controlled_predicted_sql_execution_report.json")
        metrics = _attach_predicted_sql_metrics(
            _metrics(
                _read(Path(artifacts["evaluation_dir"]) / "generic_model_evaluation_report.json"),
                _read(Path(artifacts["evaluation_dir"]) / "execution_aware_evaluation_report.json"),
            ),
            controlled_predicted_sql_report,
        )
        candidate = ModelCandidate(
            "adaptive_router",
            str(ROOT / "artifacts"),
            "adaptive_router",
            metrics,
            datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            {"controlled_predicted_sql_report": controlled_predicted_sql_report},
        )
        report = ModelSelector().select_best([candidate], load_thresholds(ROOT / "evaluation/model_quality_thresholds.yaml"))
        SelectionReporter().write(Path(artifacts["evaluation_dir"]) / "model_selection_report.json", report)
        return {"status": "completed", "summary": {"selected": bool(report.get("selected_model")), "blocking": report.get("blocking_issues", [])}}

    def _run_promote_model_if_better(self, config: PipelineConfig) -> dict[str, Any]:
        if config.smoke:
            return {"status": "skipped", "reason": "promotion skipped in smoke pipeline"}
        return {"status": "skipped", "reason": "run training/promote_model_if_better.py explicitly after review"}

    def _run_build_semantic_profile(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import write_json
        from semantic_layer import build_semantic_profile
        from semantic_layer.semantic_profile_store import SemanticProfileStore

        schema = _connected_schema(config)
        artifacts = _artifacts(config)
        schema_path = Path(artifacts["schema_dir"]) / "current_schema.json"
        write_json(schema_path, schema)
        profile = build_semantic_profile(schema)
        SemanticProfileStore(ROOT / "artifacts/semantic_profiles").save(profile["schema_fingerprint"], profile)
        return {"status": "completed", "summary": {"schema": str(schema_path), "tables": len(profile.get("tables") or {})}}

    def _run_generate_connected_db_regressions(self, config: PipelineConfig) -> dict[str, Any]:
        from connected_db_testing.schema_case_generator import SchemaCaseGenerator, write_cases_jsonl

        schema = _connected_schema(config)
        output = Path(_artifacts(config)["connected_db_regression_dir"]) / "generated_cases.jsonl"
        cases = SchemaCaseGenerator().generate_cases(schema, max_tables=3 if config.smoke else None)
        write_cases_jsonl(str(output), cases)
        return {"status": "completed", "summary": {"case_count": len(cases), "output": str(output)}}

    def _run_run_connected_db_regressions(self, config: PipelineConfig) -> dict[str, Any]:
        from connected_db_testing.generated_case_runner import ConnectedDBRegressionReporter, ConnectedDBRegressionRunner
        from dataset_training.utils import read_jsonl

        schema = _connected_schema(config)
        artifacts = _artifacts(config)
        cases_path = Path(artifacts["connected_db_regression_dir"]) / "generated_cases.jsonl"
        if not cases_path.exists():
            self._run_generate_connected_db_regressions(config)
        report = ConnectedDBRegressionRunner().run(read_jsonl(cases_path), schema)
        ConnectedDBRegressionReporter().write(report, Path(artifacts["connected_db_regression_dir"]) / "regression_report.json")
        return {"status": "completed", "summary": report["summary"]}

    def _run_run_app_smoke_check(self, config: PipelineConfig) -> dict[str, Any]:
        if config.skip_heavy_steps:
            return {"status": "completed", "summary": {"skipped": True, "reason": "skip_heavy_steps enabled"}}
        from model_bundle.bundle_loader import ModelBundleLoader
        from nl2sql_v1.schema import ColumnInfo, SchemaGraph, TableInfo
        from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel

        integrated = config.training.get("_integrated_config") or {}
        candidates = [
            _candidate_bundle_dir(config),
            ROOT / integrated.get("paths", {}).get("current_bundle_dir", "artifacts/model_bundle/current"),
        ]
        bundle_dir = next((path for path in candidates if (path / "bundle_manifest.json").exists()), None)
        if bundle_dir is None:
            return {"status": "failed", "error": "No candidate/current model bundle available for app runtime smoke"}
        bundle = ModelBundleLoader().load(bundle_dir)
        # Load model from bundle directory (not separate retrieval/neural dirs)
        model = RetrievalNL2SQLModel.load(
            artifact_dir=str(bundle_dir),
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
            "service_orders": table("service_orders", {"id": "integer", "cost": "numeric", "status": "text", "created_at": "timestamp"}),
        }, dialect="sqlite")
        issues = []
        results = []
        for question in ["list all users", "show service orders", "show users where role is admin"]:
            result = model.predict(question, schema)
            validation = result.validation or {}
            clarified = bool(result.needs_clarification or result.clarification_questions)
            if (validation.get("is_valid") is False or validation.get("ok") is False) and not clarified:
                issues.append(f"invalid prediction for {question!r}: {validation}")
            if result.raw_confidence is None or result.calibrated_confidence is None:
                issues.append(f"missing confidence fields for {question!r}")
            results.append({
                "question": question,
                "source_model": result.source_model,
                "sql_present": bool(result.sql),
                "clarification": clarified,
                "raw_confidence": result.raw_confidence,
                "calibrated_confidence": result.calibrated_confidence,
                "abstain": result.abstain,
            })
        calibration_report = Path(bundle["evaluation_dir"]) / "calibration_report.json"
        calibration_loaded = bool(model.orchestrator.confidence_calibration)
        if calibration_report.exists() and not calibration_loaded:
            issues.append("calibration report exists but runtime did not load it")
        if model.artifact_dir is None:
            issues.append("runtime fell back to sample/dev artifacts")

        # Controlled abstention test: temporarily inject a very high threshold
        # to prove the calibration is behaviorally active (not just loaded).
        abstention_tested = False
        if calibration_loaded and model.orchestrator.confidence_calibration:
            saved_threshold = model.orchestrator.confidence_calibration.get("conformal_confidence_threshold")
            try:
                model.orchestrator.confidence_calibration["conformal_confidence_threshold"] = 0.99
                abstention_result = model.predict("list all users", schema)
                if not abstention_result.abstain:
                    issues.append(
                        "Abstention test failed: with conformal_threshold=0.99, "
                        "runtime should abstain but returned abstain=False"
                    )
                abstention_tested = True
            except Exception as exc:
                issues.append(f"Abstention test error: {exc}")
            finally:
                if saved_threshold is not None:
                    model.orchestrator.confidence_calibration["conformal_confidence_threshold"] = saved_threshold
                else:
                    model.orchestrator.confidence_calibration.pop("conformal_confidence_threshold", None)

        # Bundle metadata assertions
        runtime_source = getattr(model, "runtime_source", None) or "unknown"
        bundle_id = getattr(model, "bundle_id", None) or ""
        bundle_status = getattr(model, "bundle_status", None) or ""
        dev_fallback_used = getattr(model, "dev_fallback_used", False)
        if dev_fallback_used:
            issues.append("runtime used dev fallback instead of bundle")

        summary = {
            "bundle_dir": str(bundle_dir),
            "bundle_id": bundle_id,
            "bundle_status": bundle_status,
            "runtime_source": runtime_source,
            "dev_fallback_used": dev_fallback_used,
            "calibration_loaded": calibration_loaded,
            "abstention_behavior_verified": abstention_tested,
            "predictions": results,
        }
        if issues:
            return {"status": "failed", "error": "; ".join(issues), "summary": summary}
        return {"status": "completed", "summary": summary}

    def _run_run_quality_gate(self, config: PipelineConfig) -> dict[str, Any]:
        from quality_gates.model_quality_gate import ModelQualityGate
        from quality_gates.thresholds import load_thresholds

        artifacts = _artifacts(config)
        integrated = config.training.get("_integrated_config") or {}
        thresholds_path = integrated.get("quality_gate", {}).get("thresholds", "evaluation/model_quality_thresholds.yaml")
        thresholds = load_thresholds(ROOT / thresholds_path)

        eval_path = Path(artifacts["evaluation_dir"]) / "generic_model_evaluation_report.json"
        if not eval_path.exists():
            return {"status": "failed", "error": f"Missing evaluation report: {eval_path}"}
        eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
        gate_cfg = integrated.get("quality_gate", {}) or {}
        gate_mode = str(gate_cfg.get("mode") or ("production" if gate_cfg.get("required", False) and not config.smoke else "smoke"))
        eval_report["quality_gate_mode"] = gate_mode
        eval_report["pipeline_name"] = config.pipeline_name
        eval_report["allow_intent_accuracy_simple_query_fallback"] = gate_mode not in {"production", "full"}
        execution_path = Path(artifacts["evaluation_dir"]) / "execution_aware_evaluation_report.json"
        if execution_path.exists():
            execution_report = json.loads(execution_path.read_text(encoding="utf-8"))
            execution_summary = execution_report.get("summary") or {}
            if "execution_match_rate" in execution_summary:
                eval_report.setdefault("summary", {})["execution_match_rate"] = execution_summary["execution_match_rate"]
            eval_report["execution_aware_evaluation"] = {
                **execution_report,
                "enabled": True,
                "required": bool(integrated.get("evaluation", {}).get("run_execution_aware", False)),
            }
        else:
            required = bool(integrated.get("evaluation", {}).get("run_execution_aware", False))
            eval_report["execution_aware_evaluation"] = {
                "enabled": False,
                "required": required,
                "reason": "disabled by config" if not required else f"missing report: {execution_path}",
            }
        predicted_sql_path = Path(artifacts["evaluation_dir"]) / "controlled_predicted_sql_execution_report.json"
        controlled_predicted_cfg = (integrated.get("execution_aware") or {}).get("controlled_predicted_sql") or {}
        if predicted_sql_path.exists():
            eval_report["controlled_predicted_sql_execution"] = json.loads(predicted_sql_path.read_text(encoding="utf-8"))
            eval_report["controlled_predicted_sql_required"] = bool(
                controlled_predicted_cfg.get("required_for_full_training", False)
            )
        elif controlled_predicted_cfg.get("enabled", False):
            eval_report["controlled_predicted_sql_execution"] = {
                "available": False,
                "required": bool(controlled_predicted_cfg.get("required_for_full_training", False)),
            }
        contribution_path = ROOT / "artifacts/generic_training/dataset_contribution_report.json"
        eval_report["dataset_contribution_report_required"] = True
        if contribution_path.exists():
            eval_report["dataset_contribution_report"] = json.loads(contribution_path.read_text(encoding="utf-8"))
        gold_path = Path(artifacts["self_training_dir"]) / "validation_gold_comparison_report.json"
        if gold_path.exists():
            gold_report = json.loads(gold_path.read_text(encoding="utf-8"))
            gold_summary = gold_report.get("summary") or {}
            if "gold_comparison_score" in gold_summary:
                eval_report.setdefault("summary", {})["gold_comparison_score"] = gold_summary["gold_comparison_score"]

        report = ModelQualityGate().evaluate(eval_report, thresholds)
        output = Path(artifacts["evaluation_dir"]) / "model_quality_gate_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if not report.get("passed") and integrated.get("quality_gate", {}).get("required", False):
            return {"status": "failed", "error": "Quality gate failed", "summary": report}
        return {"status": "completed", "summary": {"passed": report["passed"], "failed_checks": len(report.get("failed_checks", []))}}

    def _run_build_model_bundle(self, config: PipelineConfig) -> dict[str, Any]:
        from model_bundle.bundle_builder import ModelBundleBuilder

        integrated = config.training.get("_integrated_config") or {}
        artifacts = _artifacts(config)
        eval_path = Path(artifacts["evaluation_dir"]) / "generic_model_evaluation_report.json"
        qg_path = Path(artifacts["evaluation_dir"]) / "model_quality_gate_report.json"
        pipeline_path = ROOT / "artifacts/pipeline/train_model_report.json"
        live_pipeline_path = ROOT / "artifacts/pipeline/pipeline_report.json"
        pipeline_report = {}
        if pipeline_path.exists():
            pipeline_report = json.loads(pipeline_path.read_text(encoding="utf-8"))
        elif live_pipeline_path.exists():
            pipeline_report = json.loads(live_pipeline_path.read_text(encoding="utf-8"))
        result = ModelBundleBuilder().build_candidate_bundle(
            work_dir=ROOT / "artifacts",
            output_dir=_candidate_bundle_dir(config),
            config=integrated,
            pipeline_report=pipeline_report,
            evaluation_report=json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None,
            quality_gate_report=json.loads(qg_path.read_text(encoding="utf-8")) if qg_path.exists() else None,
        )
        return {"status": "completed", "summary": result}

    def _run_validate_model_bundle(self, config: PipelineConfig) -> dict[str, Any]:
        from model_bundle.bundle_manifest import load_manifest, save_manifest
        from model_bundle.bundle_validator import ModelBundleValidator

        candidate = _candidate_bundle_dir(config)
        result = ModelBundleValidator().validate(candidate, config=config.training.get("_integrated_config") or {})
        (candidate / "bundle_validation_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_path = candidate / "bundle_manifest.json"
        if manifest_path.exists() and result.get("lifecycle_proof"):
            manifest = load_manifest(manifest_path)
            manifest.lifecycle_proof = {**(manifest.lifecycle_proof or {}), **(result.get("lifecycle_proof") or {})}
            save_manifest(manifest, manifest_path)
        if not result.get("passed"):
            return {"status": "failed", "error": "; ".join(result.get("blocking_issues", [])), "summary": result}
        return {"status": "completed", "summary": result}

    def _run_promote_model_bundle(self, config: PipelineConfig) -> dict[str, Any]:
        from model_bundle.bundle_promoter import ModelBundlePromoter

        integrated = config.training.get("_integrated_config") or {}
        current_dir = ROOT / integrated.get("paths", {}).get("current_bundle_dir", "artifacts/model_bundle/current")
        skip_qg = not integrated.get("quality_gate", {}).get("required", False)
        result = ModelBundlePromoter().promote(_candidate_bundle_dir(config), current_dir, skip_quality_gate=skip_qg)
        return {"status": "completed" if result.get("promoted") or skip_qg else "failed", "summary": result, "error": result.get("reason")}


def _canonical_step(step: str) -> str:
    return STEP_ALIASES.get(step, step)


def _candidate_bundle_dir(config: PipelineConfig) -> Path:
    integrated = config.training.get("_integrated_config") or {}
    return ROOT / integrated.get("paths", {}).get("candidate_bundle_dir", "artifacts/model_bundle/candidate")


def _artifacts(config: PipelineConfig) -> dict[str, str]:
    defaults = {
        "generic_training_dir": str(ROOT / "artifacts/generic_training"),
        "retrieval_model_dir": str(ROOT / "artifacts/work/retrieval_ir"),
        "neural_model_dir": str(ROOT / "artifacts/work/neural_ir"),
        "adaptive_ranker_dir": str(ROOT / "artifacts/work/adaptive_ranker"),
        "self_training_dir": str(ROOT / "artifacts/self_training"),
        "evaluation_dir": str(ROOT / "artifacts/work/evaluation"),
        "schema_dir": str(ROOT / "artifacts/schema"),
        "connected_db_regression_dir": str(ROOT / "artifacts/connected_db_regressions"),
        "candidate_bundle_dir": str(ROOT / "artifacts/model_bundle/candidate"),
        "current_bundle_dir": str(ROOT / "artifacts/model_bundle/current"),
        "bundle_dir": "",
        "calibration_report_path": str(ROOT / "artifacts/work/evaluation/calibration_report.json"),
    }
    return {**defaults, **{key: str(value) for key, value in config.artifacts.items()}}


def _enforce_dataset_contribution(report: dict[str, Any], config: PipelineConfig) -> None:
    contribution = report.get("dataset_contribution_report") or {}
    integrated = config.training.get("_integrated_config") or {}
    allow_missing = bool(integrated.get("datasets", {}).get("allow_missing_dataset", False))
    if config.smoke or allow_missing:
        return
    minimum_failures = contribution.get("minimum_failures") or []
    if minimum_failures:
        details = [
            f"{item.get('dataset')}={item.get('converted_to_queryir', 0)}"
            f"/{item.get('minimum_required', 0)}"
            for item in minimum_failures
        ]
        raise ValueError(
            "Requested datasets did not meet minimum QueryIR contribution: "
            + ", ".join(details)
            + ". Set lower datasets.min_converted_examples_required values only for explicit dev runs."
        )
    requested = [str(item) for item in contribution.get("datasets_requested") or config.datasets.get("names", [])]
    blocking = []
    for name in requested:
        if name == "bird-full":
            continue
        row = (contribution.get("by_dataset") or {}).get(name) or {}
        if int(row.get("converted_to_queryir", 0)) <= 0:
            blocking.append(name)
    if blocking:
        raise ValueError(
            "Requested datasets produced zero usable QueryIR examples: "
            + ", ".join(blocking)
            + ". Set datasets.allow_missing_dataset=true only for explicit dev/smoke runs."
        )


def _connected_schema(config: PipelineConfig) -> dict[str, Any]:
    artifacts = _artifacts(config)
    schema_path = Path(config.training.get("connected_schema_path") or Path(artifacts["schema_dir"]) / "current_schema.json")
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    return {
        "dialect": "postgres",
        "database": "connected_smoke",
        "schema_name": "public",
        "tables": {
            "users": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "name": {"type": "text"},
                    "role": {"type": "text"},
                    "status": {"type": "text"},
                    "password_hash": {"type": "text"},
                    "created_at": {"type": "timestamp"},
                }
            },
            "berth_masters": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "berth_code": {"type": "text"},
                    "berth_name": {"type": "text"},
                    "status": {"type": "text"},
                    "created_at": {"type": "timestamp"},
                }
            },
            "assignments": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "user_id": {"type": "integer"},
                    "berth_id": {"type": "integer"},
                    "status": {"type": "text"},
                    "assigned_date": {"type": "date"},
                }
            },
        },
        "relationships": [
            {"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"},
            {"from_table": "assignments", "from_column": "berth_id", "to_table": "berth_masters", "to_column": "id"},
        ],
    }
