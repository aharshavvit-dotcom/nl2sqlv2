from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline_config import PipelineConfig


ROOT = Path(__file__).resolve().parents[1]


class StepRunner:
    def run_step(self, step: str, config: PipelineConfig) -> dict[str, Any]:
        if step in {"build_generic_ir_corpus", "build_retrieval_rag_index", "train_neural_ir_model"} and config.skip_heavy_steps:
            return {"status": "skipped", "reason": "skip_heavy_steps enabled for smoke pipeline"}
        method = getattr(self, f"_run_{step}", None)
        if method is None:
            return {"status": "skipped", "reason": f"no runner implemented for {step}"}
        return method(config)

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
        validation = Path(config.training.get("validation_path") or ROOT / "data/processed/generic_ir_validation.jsonl")
        args = _Args(
            input=validation,
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

        artifacts = _artifacts(config)
        predictions = Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl"
        result = HardNegativeMiner().mine(read_jsonl(predictions))
        output = ROOT / "data/processed/self_training"
        write_jsonl(output / "mined_hard_negatives.jsonl", result["mined_hard_negatives"])
        write_json(output / "error_summary.json", result["error_summary"])
        return {"status": "completed", "summary": result["error_summary"]}

    def _run_build_corrections_from_gold(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import read_jsonl, write_json, write_jsonl
        from self_training.correction_builder import CorrectionBuilder

        artifacts = _artifacts(config)
        predictions = Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl"
        result = CorrectionBuilder().build(read_jsonl(predictions))
        output = ROOT / "data/processed/self_training"
        write_jsonl(output / "correction_positive_examples.jsonl", result["correction_positive_examples"])
        write_jsonl(output / "queryir_repair_examples.jsonl", result["queryir_repair_examples"])
        write_json(output / "correction_summary.json", result["summary"])
        return {"status": "completed", "summary": result["summary"]}

    def _run_train_ranking_from_gold(self, config: PipelineConfig) -> dict[str, Any]:
        from dataset_training.utils import read_jsonl
        from self_training.ranking_trainer import RankingTrainer

        artifacts = _artifacts(config)
        report = RankingTrainer().train(read_jsonl(Path(artifacts["self_training_dir"]) / "validation_predictions.jsonl"), ROOT / "artifacts/adaptive_ranker")
        return {"status": "completed", "summary": report}

    def _run_run_self_improvement_loop(self, config: PipelineConfig) -> dict[str, Any]:
        from self_training.self_improvement_loop import SelfImprovementLoop

        artifacts = _artifacts(config)
        report = SelfImprovementLoop().run(
            train_path=config.training.get("train_path") or ROOT / "data/processed/generic_ir_train.jsonl",
            validation_path=config.training.get("validation_path") or ROOT / "data/processed/generic_ir_validation.jsonl",
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

    def _run_evaluate_generic_models(self, config: PipelineConfig) -> dict[str, Any]:
        if config.skip_heavy_steps:
            return {"status": "skipped", "reason": "existing generic evaluation report reused in smoke pipeline"}
        return {"status": "skipped", "reason": "invoke training/evaluate_generic_models.py for full evaluation"}

    def _run_select_best_model(self, config: PipelineConfig) -> dict[str, Any]:
        from training.select_best_model import _metrics, _read
        from model_selection.model_candidate import ModelCandidate
        from model_selection.model_selector import ModelSelector
        from model_selection.selection_reporter import SelectionReporter
        from quality_gates.thresholds import load_thresholds
        from datetime import datetime, timezone

        artifacts = _artifacts(config)
        metrics = _metrics(_read(Path(artifacts["evaluation_dir"]) / "generic_model_evaluation_report.json"), _read(Path(artifacts["evaluation_dir"]) / "execution_aware_evaluation_report.json"))
        candidate = ModelCandidate("adaptive_router", str(ROOT / "artifacts"), "adaptive_router", metrics, datetime.now(timezone.utc).replace(microsecond=0).isoformat(), {})
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
        return {
            "status": "completed",
            "summary": {
                "schema": str(schema_path),
                "tables": len(profile.get("tables") or {}),
                "metrics": len(profile.get("metrics") or {}),
                "dimensions": len(profile.get("dimensions") or {}),
            },
        }

    def _run_generate_connected_db_regressions(self, config: PipelineConfig) -> dict[str, Any]:
        from connected_db_testing.schema_case_generator import SchemaCaseGenerator, write_cases_jsonl

        schema = _connected_schema(config)
        artifacts = _artifacts(config)
        output = Path(artifacts["connected_db_regression_dir"]) / "generated_cases.jsonl"
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
        cases = read_jsonl(cases_path)
        report = ConnectedDBRegressionRunner().run(cases, schema)
        output = Path(artifacts["connected_db_regression_dir"]) / "regression_report.json"
        ConnectedDBRegressionReporter().write(report, output)
        return {"status": "completed", "summary": report["summary"]}

    def _run_run_app_smoke_check(self, config: PipelineConfig) -> dict[str, Any]:
        return {"status": "completed", "summary": {"streamlit_app": str(ROOT / "app/streamlit_app.py"), "exists": (ROOT / "app/streamlit_app.py").exists()}}


def _artifacts(config: PipelineConfig) -> dict[str, str]:
    defaults = {
        "generic_training_dir": str(ROOT / "artifacts/generic_training"),
        "retrieval_model_dir": str(ROOT / "artifacts/retrieval_ir_model"),
        "neural_model_dir": str(ROOT / "artifacts/neural_ir_model"),
        "self_training_dir": str(ROOT / "artifacts/self_training"),
        "evaluation_dir": str(ROOT / "artifacts/evaluation"),
        "schema_dir": str(ROOT / "artifacts/schema"),
        "connected_db_regression_dir": str(ROOT / "artifacts/connected_db_regressions"),
    }
    return {**defaults, **{key: str(value) for key, value in config.artifacts.items()}}


def _connected_schema(config: PipelineConfig) -> dict[str, Any]:
    artifacts = _artifacts(config)
    schema_path = Path(config.training.get("connected_schema_path") or Path(artifacts["schema_dir"]) / "current_schema.json")
    if schema_path.exists():
        import json

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
