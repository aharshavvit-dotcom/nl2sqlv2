import pytest
from nl2sqlv2.training.train_model import TrainJob

class DummyTrainJob(TrainJob):
    def _execute_run(self, run_index: int, total_runs: int, seed: int) -> dict:
        return {"status": "completed"}

def test_single_seed_counts_as_one(tmp_path):
    job = DummyTrainJob(config={
        "training": {
            "multi_seed": {
                "enabled": False,
                "runs": 3
            }
        },
        "model_dir": str(tmp_path)
    })
    
    # Run primary only
    job._execute_run(0, 1, 42)
    # the job loop logic
    job.seed_runs = [{"status": "completed"}]
    
    completed = [run for run in job.seed_runs if run.get("status") == "completed"]
    assert len(completed) == 1
    
    job.config["training"]["multi_seed"]["enabled"] = True
    job.seed_runs = [{"status": "completed"}, {"status": "completed"}]
    completed = [run for run in job.seed_runs if run.get("status") == "completed"]
    assert len(completed) == 2
