from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_selection.champion_challenger import ChampionChallengerRegistry
from model_selection.promotion_policy import PromotionPolicy
from quality_gates.thresholds import load_thresholds
from training.select_best_model import _metrics, _read


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model artifact if it beats the current champion.")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "artifacts" / "neural_ir_model")
    parser.add_argument("--model-name", default="neural_ir_model")
    parser.add_argument("--evaluation-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "generic_model_evaluation_report.json")
    parser.add_argument("--execution-report", type=Path, default=ROOT / "artifacts" / "evaluation" / "execution_aware_evaluation_report.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evaluation" / "model_quality_thresholds.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "model_registry" / "promotion_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = ChampionChallengerRegistry()
    metrics = _metrics(_read(args.evaluation_report), _read(args.execution_report))
    challenger = registry.register_challenger(args.model_name, str(args.candidate_dir), metrics)
    champion = registry.get_current_champion(args.model_name)
    decision = PromotionPolicy().can_promote(metrics, (champion or {}).get("metrics") if champion else None, load_thresholds(args.thresholds))
    promoted = None
    if decision["can_promote"]:
        promoted = registry.promote_challenger(args.model_name, challenger["challenger_id"])
    report = {"challenger": challenger, "current_champion_before": champion, "promotion_decision": decision, "promoted": promoted}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"promoted": promoted is not None, "blocking_issues": decision["blocking_issues"]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
