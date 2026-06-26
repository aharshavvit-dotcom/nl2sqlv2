"""Canonical integrated training command.

This is the ONE primary training command for the NL-to-SQL system.
It orchestrates corpus building, retrieval/neural training, evaluation,
quality gates, and model bundle creation.

Usage:
    python training/train_model.py --config configs/training.yaml
    python training/train_model.py --config configs/smoke_training.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical NL-to-SQL integrated training command.",
        epilog="This is the primary training entry point. "
               "For advanced commands, see docs/developer_commands.md.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to training config YAML (e.g. configs/training.yaml)")
    parser.add_argument("--start-at", type=str, default=None,
                        help="Resume pipeline from this step")
    parser.add_argument("--stop-after", type=str, default=None,
                        help="Stop pipeline after this step")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed steps")
    parser.add_argument("--force", action="store_true",
                        help="Continue past input/output validation failures")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    return parser.parse_args()


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the training config."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = str(config_path)
    return config


def validate_environment(config: dict[str, Any]) -> list[str]:
    """Check that the environment is ready for training."""
    issues = []
    # Check processed dir exists or can be created
    processed_dir = ROOT / config.get("paths", {}).get("processed_dir", "data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check artifacts dir
    artifacts_dir = ROOT / config.get("paths", {}).get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return issues


def verify_datasets(config: dict[str, Any]) -> bool:
    """Verify that required datasets are available."""
    dataset_names = config.get("datasets", {}).get("names", [])
    if not dataset_names:
        print("Warning: No datasets specified in config.")
        return True

    try:
        from scripts.verify_datasets import verify_all
        rows = {row.dataset: row for row in verify_all()}
        dataset_map = {"wikisql": "WikiSQL", "spider": "Spider", "bird-mini": "BIRD Mini-Dev"}
        missing = []
        for name in dataset_names:
            display_name = dataset_map.get(name, name)
            if display_name in rows and not rows[display_name].ready:
                missing.append(name)
        if missing:
            print(f"Warning: Datasets not ready: {missing}")
            print("Run: python scripts/download_datasets.py --datasets " + " ".join(missing))
            return False
        return True
    except Exception as exc:
        print(f"Warning: Could not verify datasets: {exc}")
        return True


def allow_missing_datasets(config: dict[str, Any]) -> bool:
    datasets = config.get("datasets", {}) or {}
    pipeline = config.get("pipeline", {}) or {}
    if bool(datasets.get("allow_missing_dataset", False)):
        return True
    if str(pipeline.get("name", "")).lower().startswith("smoke"):
        return True
    return int(datasets.get("max_examples", 5000) or 5000) <= 200


from orchestration.pipeline_config import build_pipeline_steps


def config_to_pipeline_config(config: dict[str, Any], steps: list[str]) -> dict[str, Any]:
    """Convert integrated training config to PipelineConfig-compatible dict."""
    pipeline = config.get("pipeline", {})
    paths = config.get("paths", {})
    neural = config.get("neural", {})
    datasets = config.get("datasets", {})
    self_training = config.get("self_training", {})
    evaluation = config.get("evaluation", {})
    quality_gate = config.get("quality_gate", {})
    evaluation_dir = ROOT / evaluation.get("output_dir", "artifacts/evaluation")
    candidate_bundle_dir = ROOT / paths.get("candidate_bundle_dir", "artifacts/model_bundle/candidate")
    current_bundle_dir = ROOT / paths.get("current_bundle_dir", "artifacts/model_bundle/current")

    # Map to the existing PipelineConfig format
    return {
        "pipeline_name": pipeline.get("name", "integrated_training"),
        "seed": pipeline.get("seed", 42),
        "smoke": datasets.get("max_examples", 5000) <= 200,
        "skip_heavy_steps": False,
        "datasets": {
            "names": datasets.get("names", []),
            "max_examples": datasets.get("max_examples", 5000),
            "max_examples_per_dataset": datasets.get("max_examples_per_dataset", {}),
            "min_converted_examples_required": datasets.get("min_converted_examples_required", {}),
        },
        "training": {
            "neural_epochs": neural.get("epochs", 5),
            "batch_size": neural.get("batch_size", 8),
            "neural_config": neural.get("config", "configs/neural_training_default.yaml"),
            "self_improvement_iterations": self_training.get("iterations", 1),
            "max_self_training_examples": self_training.get("max_examples", 1000),
        },
        "artifacts": {
            "generic_training_dir": str(ROOT / "artifacts/generic_training"),
            "retrieval_model_dir": str(ROOT / (config.get("retrieval", {}).get("output_dir", "artifacts/retrieval_ir_model"))),
            "neural_model_dir": str(ROOT / neural.get("output_dir", "artifacts/neural_ir_model")),
            "adaptive_ranker_dir": str(ROOT / (config.get("ranker", {}).get("output_dir", "artifacts/work/adaptive_ranker"))),
            "self_training_dir": str(ROOT / "artifacts/self_training"),
            "evaluation_dir": str(evaluation_dir),
            "schema_dir": str(ROOT / "artifacts/schema"),
            "connected_db_regression_dir": str(ROOT / "artifacts/connected_db_regressions"),
            "candidate_bundle_dir": str(candidate_bundle_dir),
            "current_bundle_dir": str(current_bundle_dir),
            "bundle_dir": str(candidate_bundle_dir),
            "calibration_report_path": str(evaluation_dir / "calibration_report.json"),
        },
        "steps": steps,
        # Extended fields for new steps
        "_integrated_config": config,
    }


def run_pipeline(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run the training pipeline."""
    steps = build_pipeline_steps(config)
    pipeline_config_dict = config_to_pipeline_config(config, steps)

    # Write pipeline config for the runner
    pipeline_config_path = ROOT / "artifacts" / "pipeline" / "_current_pipeline_config.yaml"
    pipeline_config_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config_path.write_text(
        yaml.dump(pipeline_config_dict, default_flow_style=False), encoding="utf-8"
    )

    from orchestration.pipeline_runner import PipelineRunner
    runner = PipelineRunner(state_path=ROOT / "artifacts" / "pipeline" / "pipeline_state.json")
    report = runner.run(
        str(pipeline_config_path),
        start_at=args.start_at,
        stop_after=args.stop_after,
        resume=args.resume,
        force=args.force,
        dry_run=args.dry_run,
    )
    return report


def write_training_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    """Write the final training report."""
    output_dir = ROOT / "artifacts" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    enriched = {
        **report,
        "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config_path": config.get("_config_path", ""),
    }
    (output_dir / "train_model_report.json").write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# Training Model Report",
        "",
        f"**Pipeline:** {report.get('pipeline_name', 'unknown')}",
        f"**Status:** {report.get('status', 'unknown')}",
        f"**Config:** {config.get('_config_path', 'unknown')}",
        f"**Completed:** {enriched['completed_at']}",
        "",
        "## Pipeline Steps",
        "",
    ]
    for step in report.get("steps", []):
        status_icon = {"completed": "[ok]", "failed": "[failed]", "skipped": "[skipped]", "dry_run": "[dry-run]"}.get(
            step.get("status", ""), "[unknown]"
        )
        lines.append(f"- {status_icon} **{step.get('step', 'unknown')}**: {step.get('status', 'unknown')}")
        if step.get("error"):
            lines.append(f"  - Error: {step['error']}")
        if step.get("reason"):
            lines.append(f"  - Reason: {step['reason']}")
    lines.append("")
    (output_dir / "train_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    print(f"{'=' * 60}")
    print(f"NL-to-SQL Integrated Training Pipeline")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print()

    # 1. Load config
    config = load_training_config(args.config)
    pipeline_name = config.get("pipeline", {}).get("name", "integrated_training")
    print(f"Pipeline: {pipeline_name}")

    # 2. Validate environment
    env_issues = validate_environment(config)
    if env_issues:
        print(f"Environment issues: {env_issues}")
        if config.get("pipeline", {}).get("fail_fast", True):
            return 1

    # 3. Verify datasets
    datasets_ok = verify_datasets(config)
    if not datasets_ok:
        if not allow_missing_datasets(config):
            print("Error: Required datasets are not ready for full training.")
            print("Full training will not continue with missing required datasets.")
            return 1
        print("Warning: Some datasets are not ready. Continuing because this is an explicit smoke/dev run.")

    # 4. Show plan in dry-run mode
    if args.dry_run:
        steps = build_pipeline_steps(config)
        print(f"\nDry-run: would execute {len(steps)} steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print()

    # 5. Run pipeline
    print(f"\nStarting pipeline...")
    report = run_pipeline(config, args)

    # 6. Write training report
    if not args.dry_run:
        write_training_report(report, config)

    # 7. Summary (compute status BEFORE multi-seed block uses it)
    status = report.get("status", "unknown")
    step_count = len(report.get("steps", []))
    completed = sum(1 for s in report.get("steps", []) if s.get("status") == "completed")
    failed = sum(1 for s in report.get("steps", []) if s.get("status") == "failed")
    skipped = sum(1 for s in report.get("steps", []) if s.get("status") == "skipped")

    # 7b. Multi-seed variance analysis (runs only when enabled and primary run succeeded)
    seeds_config = config.get("seeds", {})
    if seeds_config.get("enabled", False) and not args.dry_run and status == "completed":
        seed_values = seeds_config.get("values", [42])
        tracked_metrics = seeds_config.get("metrics", [
            "intent_macro_f1", "base_table_accuracy", "sql_validation_rate",
            "query_ir_validity_rate", "execution_match_rate",
        ])
        report_output = ROOT / seeds_config.get("report_output", "artifacts/evaluation/multi_seed_variance_report.json")
        print(f"\n  Multi-seed variance analysis ({len(seed_values)} seeds)")
        print(f"  Seeds: {seed_values}")
        print(f"  Tracking: {tracked_metrics}")
        variance_report = _run_multi_seed_variance(
            config, report, seed_values, tracked_metrics, report_output,
        )
        if variance_report:
            high_var = variance_report.get("high_variance_metrics", [])
            if high_var:
                print(f"  ⚠ High variance detected in: {high_var}")
            else:
                print(f"  ✓ All metrics within acceptable variance")

    print(f"\n{'=' * 60}")
    print(f"Pipeline: {status.upper()}")
    print(f"Steps: {completed} completed, {failed} failed, {skipped} skipped (of {step_count})")

    if status == "completed":
        print(f"\nReports written to:")
        print(f"  artifacts/pipeline/train_model_report.json")
        print(f"  artifacts/pipeline/train_model_report.md")
        bundle_cfg = config.get("bundle", {})
        if bundle_cfg.get("build", True):
            print(f"  artifacts/model_bundle/candidate/bundle_manifest.json")
        if bundle_cfg.get("promote_if_quality_gate_passes", False):
            print(f"  artifacts/model_bundle/current/bundle_manifest.json (if promoted)")
    elif failed > 0:
        failed_steps = [s for s in report.get("steps", []) if s.get("status") == "failed"]
        print(f"\nFailed steps:")
        for step in failed_steps:
            print(f"  - {step.get('step')}: {step.get('error', 'unknown error')}")
    print(f"{'=' * 60}")

    return 0 if status == "completed" else 1


def _run_multi_seed_variance(
    config: dict[str, Any],
    primary_report: dict[str, Any],
    seed_values: list[int],
    tracked_metrics: list[str],
    report_output: Path,
) -> dict[str, Any] | None:
    """Run multi-seed evaluation-only stability analysis.

    For each configured seed:
      1. Sets deterministic random seed (Python, NumPy, torch)
      2. Re-runs evaluation step only (not full training) via StepRunner
      3. Collects per-seed evaluation metrics

    This is evaluation-only stability, NOT true training variance.
    Full per-seed re-training is a future enhancement.
    """
    import copy
    import random
    import statistics

    import numpy as np

    # Extract primary run's metrics from evaluation step result
    primary_metrics = _extract_eval_metrics(primary_report, tracked_metrics)
    if not primary_metrics:
        print("  [Note] Could not extract evaluation metrics from primary run — skipping variance analysis.")
        return None

    # Collect per-seed metrics. Primary run counts as seed 0 (the pipeline seed).
    per_seed_metrics: dict[str, list[float]] = {metric: [] for metric in tracked_metrics}
    for metric_name, value in primary_metrics.items():
        if value is not None:
            per_seed_metrics[metric_name].append(value)
    primary_seed = int(config.get("pipeline", {}).get("seed", 42))
    resolved = config_to_pipeline_config(config, ["evaluate_generic_models"])
    resolved_artifacts = dict(resolved.get("artifacts") or {})
    model_source, model_bundle_dir = _seed_model_source(resolved_artifacts)
    seed_runs: list[dict[str, Any]] = [{
        "mode": "evaluation_only_stability",
        "seed": primary_seed,
        "status": "completed",
        "input_model_source": model_source,
        "model_bundle_dir": model_bundle_dir,
        "evaluation_output_dir": str(resolved_artifacts.get("evaluation_dir", ROOT / "artifacts/evaluation")),
        "used_primary_model_artifacts": True,
        "is_primary_pipeline_run": True,
    }]

    # Run evaluation-only re-runs for additional seeds
    additional_seeds = [s for s in seed_values if s != primary_seed]
    if additional_seeds:
        try:
            from orchestration.pipeline_config import PipelineConfig
            from orchestration.step_runner import StepRunner
            for seed_val in additional_seeds:
                print(f"  Re-evaluating with seed {seed_val}...")
                # Set deterministic seeds
                random.seed(seed_val)
                np.random.seed(seed_val)
                try:
                    import torch
                    torch.manual_seed(seed_val)
                except ImportError:
                    pass
                # Create child config with seeds disabled to prevent recursion
                child_config = copy.deepcopy(config)
                child_config.setdefault("pipeline", {})["seed"] = seed_val
                if "seeds" in child_config:
                    child_config["seeds"]["enabled"] = False  # Anti-recursion
                child_resolved = config_to_pipeline_config(child_config, ["evaluate_generic_models"])
                child_artifacts = dict(child_resolved.get("artifacts") or {})
                seed_eval_dir = Path(resolved_artifacts.get("evaluation_dir", ROOT / "artifacts/evaluation")) / "seeds" / f"seed_{seed_val}"
                child_artifacts["evaluation_dir"] = str(seed_eval_dir)
                child_source, child_bundle_dir = _seed_model_source(child_artifacts)
                # Build proper PipelineConfig for StepRunner
                child_training = {**(child_resolved.get("training") or {}), "_integrated_config": child_config}
                child_pipeline_config = PipelineConfig(
                    pipeline_name=f"seed_{seed_val}_eval",
                    seed=seed_val,
                    datasets=child_resolved.get("datasets", {}),
                    training=child_training,
                    artifacts=child_artifacts,
                    steps=["evaluate_generic_models"],
                    smoke=bool(child_resolved.get("smoke", False)),
                    integrated_config=child_config,
                )
                runner = StepRunner()
                seed_summary: dict[str, Any] = {
                    "mode": "evaluation_only_stability",
                    "seed": seed_val,
                    "input_model_source": child_source,
                    "model_bundle_dir": child_bundle_dir,
                    "evaluation_output_dir": str(seed_eval_dir),
                    "used_primary_model_artifacts": True,
                    "is_primary_pipeline_run": False,
                }
                try:
                    result = runner.run_step("evaluate_generic_models", child_pipeline_config)
                    if result.get("status") == "completed":
                        child_summary = result.get("summary") or {}
                        for metric_name in tracked_metrics:
                            val = child_summary.get(metric_name)
                            if val is not None:
                                per_seed_metrics[metric_name].append(float(val))
                        seed_summary["status"] = "completed"
                        seed_summary["metrics"] = {
                            metric: child_summary.get(metric)
                            for metric in tracked_metrics
                            if child_summary.get(metric) is not None
                        }
                    else:
                        print(f"    Seed {seed_val} evaluation did not complete: {result.get('status')}")
                        seed_summary["status"] = result.get("status", "unknown")
                        seed_summary["error"] = result.get("error")
                except Exception as exc:
                    print(f"    Seed {seed_val} evaluation failed: {exc}")
                    seed_summary["status"] = "failed"
                    seed_summary["error"] = str(exc)
                seed_runs.append(seed_summary)
        except ImportError:
            print("  [Note] StepRunner not available — falling back to single-seed baseline")

    metrics_report: dict[str, dict[str, Any]] = {}
    high_variance: list[str] = []
    metric_std_flat: dict[str, float] = {}
    metric_sample_counts: dict[str, int] = {}
    for metric in tracked_metrics:
        values = per_seed_metrics.get(metric, [])
        if not values:
            continue
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        metrics_report[metric] = {
            "values": values,
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "sample_count": len(values),
        }
        metric_std_flat[metric] = round(std_val, 6)
        metric_sample_counts[metric] = len(values)
        if std_val > 0.05:
            high_variance.append(metric)

    # Phase 4: Proper seed run accounting from actual statuses
    seed_runs_completed = sum(1 for run in seed_runs if run.get("status") == "completed")
    seed_runs_failed = sum(1 for run in seed_runs if run.get("status") == "failed")
    seed_runs_requested = len(seed_values)
    has_multi_seed = seed_runs_completed >= 2
    variance_report = {
        "enabled": True,
        "mode": "evaluation_only_stability" if has_multi_seed else "single_seed_baseline",
        "primary_seed_included": True,
        "seeds_requested": seed_values,
        "seeds_evaluated": seed_runs_completed,  # Backward compat
        "seed_runs_requested": seed_runs_requested,
        "seed_runs_completed": seed_runs_completed,
        "seed_runs_failed": seed_runs_failed,
        # Clear separation: evaluation stability vs true training variance
        "evaluation_stability_available": has_multi_seed,
        "true_training_variance_available": False,
        "is_valid_for_evaluation_stability": has_multi_seed,
        "is_valid_for_variance_governance": False,  # Evaluation-only, never true for governance
        "is_valid_for_training_variance_governance": False,  # Requires full re-training per seed
        "stochastic_inference_enabled": bool((config.get("seeds") or {}).get("stochastic_inference_enabled", False)),
        "stochastic_components": list((config.get("seeds") or {}).get("stochastic_components", [])),
        "evaluation_stability_interpretation": (
            "measures_inference_stability_under_stochastic_components"
            if bool((config.get("seeds") or {}).get("stochastic_inference_enabled", False))
            else "deterministic_path_expected_zero_variance"
        ),
        "note": (
            "Evaluation-only stability analysis across seeds. "
            "This measures prediction stability, not training variance. "
            "Full per-seed re-training is a future enhancement."
        ) if has_multi_seed else (
            "Single-seed baseline. Enable multi-seed and run full pipeline for stability analysis."
        ),
        "seed_runs": seed_runs,
        "metrics": metrics_report,
        "metric_std": metric_std_flat,  # Flat dict for backward compat with ModelSelector
        "metric_sample_counts": metric_sample_counts,
        "high_variance_metrics": high_variance,
        "passed": len(high_variance) == 0,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(variance_report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  Variance report written to: {report_output}")
    return variance_report


def _seed_model_source(artifacts: dict[str, Any]) -> tuple[str, str]:
    """Resolve which trained artifact source seed evaluations should reuse."""
    candidate = Path(str(artifacts.get("candidate_bundle_dir") or ""))
    current = Path(str(artifacts.get("current_bundle_dir") or ""))
    configured_bundle = Path(str(artifacts.get("bundle_dir") or "")) if artifacts.get("bundle_dir") else None
    preferred = configured_bundle if configured_bundle and (configured_bundle / "bundle_manifest.json").exists() else candidate
    if preferred and (preferred / "bundle_manifest.json").exists():
        artifacts["bundle_dir"] = str(preferred)
        return "model_bundle_candidate", str(preferred)
    if current and (current / "bundle_manifest.json").exists():
        artifacts["bundle_dir"] = str(current)
        return "model_bundle_current", str(current)
    artifacts["bundle_dir"] = ""
    return "artifact_dirs", ""


def _extract_eval_metrics(
    pipeline_report: dict[str, Any],
    tracked_metrics: list[str],
) -> dict[str, float | None]:
    """Extract tracked metrics from the pipeline report's evaluation step."""
    for step in pipeline_report.get("steps", []):
        if not isinstance(step, dict):
            continue
        if step.get("step") == "evaluate_generic_models" and step.get("status") == "completed":
            summary = step.get("summary") or step.get("result", {}).get("summary") or {}
            return {metric: summary.get(metric) for metric in tracked_metrics}
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
