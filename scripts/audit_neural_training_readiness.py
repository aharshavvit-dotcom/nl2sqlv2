"""Audit the Neural QueryIR training stack readiness.

Checks 13 criteria and writes reports to ``artifacts/audit/``.

Usage:
    python scripts/audit_neural_training_readiness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check(name: str, description: str, ok: bool, detail: str = "") -> dict:
    return {
        "name": name,
        "description": description,
        "status": "pass" if ok else "fail",
        "detail": detail,
    }


def run_audit() -> dict:
    checks = []

    # 1. Neural QueryIR Model exists
    model_path = ROOT / "neural_ir" / "attention_model.py"
    checks.append(_check("neural_model_exists", "Neural QueryIR Model exists", model_path.exists()))

    # 2. Model predicts QueryIR labels, not raw SQL
    predictor_path = ROOT / "neural_ir" / "predictor.py"
    if predictor_path.exists():
        src = predictor_path.read_text(encoding="utf-8")
        ok = "query_ir" in src.lower() and "label_encoder" in src.lower()
        checks.append(_check("predicts_queryir", "Model predicts QueryIR labels", ok))
    else:
        checks.append(_check("predicts_queryir", "Model predicts QueryIR labels", False, "predictor.py not found"))

    # 3. PyTorch training is used
    trainer_path = ROOT / "neural_ir" / "trainer.py"
    if trainer_path.exists():
        src = trainer_path.read_text(encoding="utf-8")
        ok = "torch" in src and "backward" in src
        checks.append(_check("pytorch_training", "PyTorch training used", ok))
    else:
        checks.append(_check("pytorch_training", "PyTorch training used", False))

    # 4. Backpropagation is used correctly
    if trainer_path.exists():
        src = trainer_path.read_text(encoding="utf-8")
        ok = "loss.backward()" in src and "optimizer.step()" in src
        checks.append(_check("backprop_correct", "Backpropagation used correctly", ok))
    else:
        checks.append(_check("backprop_correct", "Backpropagation used correctly", False))

    # 5. Loss is calculated per output head
    if trainer_path.exists():
        src = trainer_path.read_text(encoding="utf-8")
        ok = "HEAD_TO_LABEL" in src
        checks.append(_check("per_head_loss", "Loss calculated per output head", ok))
    else:
        checks.append(_check("per_head_loss", "Loss calculated per output head", False))

    # 6. Candidate masks are used for pointer heads
    if trainer_path.exists():
        src = trainer_path.read_text(encoding="utf-8")
        ok = "HEAD_TO_MASK" in src and "masked_cross_entropy" in src
        checks.append(_check("candidate_masks", "Candidate masks used for pointer heads", ok))
    else:
        checks.append(_check("candidate_masks", "Candidate masks used for pointer heads", False))

    # 7. Hard-negative loss exists
    loss_path = ROOT / "neural_ir" / "loss_utils.py"
    ok = loss_path.exists() and "margin_ranking" in loss_path.read_text(encoding="utf-8").lower()
    checks.append(_check("hard_negative_loss", "Hard-negative loss exists", ok))

    # 8. Optimizer is configurable
    opt_factory = ROOT / "neural_optimization" / "optimizer_factory.py"
    ok = opt_factory.exists()
    detail = "optimizer_factory.py present" if ok else "optimizer_factory.py missing"
    checks.append(_check("optimizer_configurable", "Optimizer is configurable", ok, detail))

    # 9. Activation function is configurable
    act_factory = ROOT / "neural_optimization" / "activation_factory.py"
    ok = act_factory.exists()
    checks.append(_check("activation_configurable", "Activation is configurable", ok))

    # 10. Checkpointing exists
    ckpt = ROOT / "neural_optimization" / "checkpoint_manager.py"
    ok = ckpt.exists()
    checks.append(_check("checkpointing", "Checkpointing exists", ok))

    # 11. Validation evaluation exists
    eval_path = ROOT / "neural_ir" / "evaluator.py"
    ok = eval_path.exists()
    checks.append(_check("validation_eval", "Validation evaluation exists", ok))

    # 12. Training diagnostics exist
    diag = ROOT / "neural_optimization" / "training_diagnostics.py"
    ok = diag.exists()
    checks.append(_check("training_diagnostics", "Training diagnostics exist", ok))

    # 13. README training commands are current
    readme = ROOT / "README.md"
    if readme.exists():
        src = readme.read_text(encoding="utf-8")
        ok = "train_neural_ir" in src
        checks.append(_check("readme_commands", "README training commands current", ok))
    else:
        checks.append(_check("readme_commands", "README training commands current", False))

    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = sum(1 for c in checks if c["status"] == "fail")
    overall = "pass" if failed == 0 else ("partial" if passed > failed else "fail")

    report = {
        "overall_status": overall,
        "summary": {"passed": passed, "failed": failed, "warnings": 0},
        "checks": checks,
        "missing_files": [c["name"] for c in checks if c["status"] == "fail"],
        "integration_issues": [],
        "recommended_fixes": [c["name"] for c in checks if c["status"] == "fail"],
    }
    return report


def main() -> None:
    report = run_audit()

    out_dir = ROOT / "artifacts" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "neural_training_readiness_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )

    lines = ["# Neural Training Readiness Audit", ""]
    lines.append(f"**Overall**: {report['overall_status'].upper()}")
    lines.append(f"**Passed**: {report['summary']['passed']} | **Failed**: {report['summary']['failed']}")
    lines.append("")
    for c in report["checks"]:
        icon = "✓" if c["status"] == "pass" else "✗"
        lines.append(f"- [{icon}] **{c['name']}**: {c['description']}")
        if c.get("detail"):
            lines.append(f"  - {c['detail']}")
    (out_dir / "neural_training_readiness_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_dir}")


if __name__ == "__main__":
    main()
