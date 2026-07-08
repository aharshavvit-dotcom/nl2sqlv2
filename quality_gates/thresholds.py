from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_THRESHOLDS = {
    "minimums": {
        "query_ir_validity_rate": 0.90,
        "sql_validation_rate": 0.90,
        "simple_query_pass_rate": 0.95,
        "no_select_star_rate": 1.00,
        "unsafe_sql_count_max": 0,
        "unnecessary_join_rate_max": 0.05,
        "wrong_table_rate_max": 0.15,
        "unseen_db_sql_validation_rate": 0.80,
        "feedback_regression_pass_rate": 0.95,
        "gold_comparison_score_min": 0.75,
        "sql_structure_match_rate_min": 0.70,
        "execution_match_rate_min": 0.60,
        "model_promotion_min_improvement": 0.01,
        "controlled_predicted_sql_execution_match_rate_min": 0.70,
        "controlled_predicted_sql_result_value_match_rate_min": 0.70,
        "controlled_predicted_sql_safe_but_wrong_sql_rate_max": 0.30,
        "controlled_predicted_sql_safe_sql_rate_min": 1.0,
        "post_abstention_unsafe_sql_count_max": 0,
        "controlled_predicted_sql_required": False,
    }
}


def load_thresholds(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return DEFAULT_THRESHOLDS
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    merged = {"minimums": {**DEFAULT_THRESHOLDS["minimums"], **(payload.get("minimums") or {})}}
    return merged
