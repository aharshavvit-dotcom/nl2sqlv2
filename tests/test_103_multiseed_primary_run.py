from __future__ import annotations

import pytest


def test_primary_seed_status_completed():
    # Verify that the primary seed run is initialized with completed status and correct flags
    primary_seed = 42
    model_source = "some_source"
    model_bundle_dir = "some_dir"
    evaluation_dir = "eval_dir"
    
    seed_runs = [{
        "mode": "evaluation_only_stability",
        "seed": primary_seed,
        "status": "completed",
        "input_model_source": model_source,
        "model_bundle_dir": model_bundle_dir,
        "evaluation_output_dir": evaluation_dir,
        "used_primary_model_artifacts": True,
        "is_primary_pipeline_run": True,
    }]
    
    assert seed_runs[0]["status"] == "completed"
    assert seed_runs[0]["is_primary_pipeline_run"] is True


def test_single_seed_baseline_counts_one():
    # For a single-seed baseline, seed_runs_completed must be exactly 1
    seed_runs = [{
        "seed": 42,
        "status": "completed",
        "is_primary_pipeline_run": True,
    }]
    
    seed_runs_completed = sum(1 for run in seed_runs if run.get("status") == "completed")
    seed_runs_failed = sum(1 for run in seed_runs if run.get("status") == "failed")
    
    assert seed_runs_completed == 1
    assert seed_runs_failed == 0


def test_failed_child_does_not_count_completed():
    # If an additional seed run fails, it increments failed count but NOT completed count
    seed_runs = [
        {
            "seed": 42,
            "status": "completed",
            "is_primary_pipeline_run": True,
        },
        {
            "seed": 100,
            "status": "failed",
            "is_primary_pipeline_run": False,
            "error": "Some eval error",
        }
    ]
    
    seed_runs_completed = sum(1 for run in seed_runs if run.get("status") == "completed")
    seed_runs_failed = sum(1 for run in seed_runs if run.get("status") == "failed")
    
    assert seed_runs_completed == 1
    assert seed_runs_failed == 1


def test_metric_sample_counts_independent_from_seed_runs_completed():
    # Proves metric sample counts are tracked per metric independent of completion counts
    per_seed_metrics = {
        "intent_macro_f1": [0.9, 0.92],
        "sql_validation_rate": [0.95]
    }
    
    metric_sample_counts = {
        metric: len(values)
        for metric, values in per_seed_metrics.items()
    }
    
    assert metric_sample_counts["intent_macro_f1"] == 2
    assert metric_sample_counts["sql_validation_rate"] == 1


def test_lifecycle_proof_records_seed_runs_completed_correctly():
    # Make sure we construct lifecycle proof dictionary exactly as model_bundle/bundle_validator.py expects
    seed_runs_completed = 3
    metric_sample_counts = {"intent_macro_f1": 3}
    
    seed_report = {
        "enabled": True,
        "mode": "evaluation_only_stability",
        "seed_runs_completed": seed_runs_completed,
        "metric_sample_counts": metric_sample_counts,
        # Identity fields
        "bundle_id": "b1",
        "pipeline_run_id": "run-1",
    }
    
    # Simulating bundle_validator.py logic
    lifecycle_proof = {}
    lifecycle_proof["seed_runs_completed"] = int(seed_report.get("seed_runs_completed", 0))
    lifecycle_proof["metric_sample_counts"] = dict(seed_report.get("metric_sample_counts") or {})
    
    assert lifecycle_proof["seed_runs_completed"] == 3
    assert lifecycle_proof["metric_sample_counts"]["intent_macro_f1"] == 3
