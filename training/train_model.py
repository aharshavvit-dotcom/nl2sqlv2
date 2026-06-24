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
            "evaluation_dir": str(ROOT / evaluation.get("output_dir", "artifacts/evaluation")),
            "schema_dir": str(ROOT / "artifacts/schema"),
            "connected_db_regression_dir": str(ROOT / "artifacts/connected_db_regressions"),
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

    # 6b. Multi-seed variance analysis (scaffolding — runs only when enabled)
    seeds_config = config.get("seeds", {})
    if seeds_config.get("enabled", False) and not args.dry_run and status == "completed":
        seed_values = seeds_config.get("values", [42])
        report_output = ROOT / seeds_config.get("report_output", "artifacts/evaluation/multi_seed_variance_report.json")
        print(f"\n  Multi-seed variance analysis enabled ({len(seed_values)} seeds)")
        print(f"  Seeds: {seed_values}")
        print(f"  Report: {report_output}")
        # TODO: Implement multi-seed iteration:
        #   for seed in seed_values:
        #       config["pipeline"]["seed"] = seed
        #       seed_report = run_pipeline(config, args)
        #       collect metrics per seed
        #   compute mean/std per metric, write variance report
        print(f"  [Note] Multi-seed iteration not yet implemented — scaffolding only.")

    # 7. Summary
    status = report.get("status", "unknown")
    step_count = len(report.get("steps", []))
    completed = sum(1 for s in report.get("steps", []) if s.get("status") == "completed")
    failed = sum(1 for s in report.get("steps", []) if s.get("status") == "failed")
    skipped = sum(1 for s in report.get("steps", []) if s.get("status") == "skipped")

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


if __name__ == "__main__":
    raise SystemExit(main())
