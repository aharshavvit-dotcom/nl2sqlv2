from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression

from .candidate_generator import GoldCandidateGenerator
from .dataset_reward_scorer import DatasetRewardScorer


FEATURE_NAMES = ["reward_score", "intent_match", "base_table_match", "join_count_delta", "metric_count_delta", "filter_count_delta"]


class RankingTrainer:
    def train(self, rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
        generator = GoldCandidateGenerator()
        scorer = DatasetRewardScorer()
        x: list[list[float]] = []
        y: list[int] = []
        for row in rows:
            gold = {"query_ir": row.get("gold_query_ir") or row.get("query_ir") or {}}
            for candidate in generator.generate(row):
                features = scorer.features(candidate, gold, row.get("schema") or {})
                x.append([float(features[name]) for name in FEATURE_NAMES])
                y.append(int(candidate.get("label", 0)))
        if len(set(y)) < 2:
            x.extend([[0.0 for _ in FEATURE_NAMES], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
            y.extend([0, 1])
        model = LogisticRegression(max_iter=200).fit(x, y)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, output / "ranker.pkl")
        report = {"training_rows": len(rows), "candidates": len(x), "positive_labels": sum(y), "feature_names": FEATURE_NAMES}
        (output / "ranking_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
