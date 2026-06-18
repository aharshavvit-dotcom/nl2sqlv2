from __future__ import annotations

from typing import Any


CURRICULUM_PHASES = {
    1: {"show_records", "count_records", "simple_filter"},
    2: {"metric_summary", "metric_by_dimension", "count_by_dimension"},
    3: {"top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "trend_by_date"},
    4: {"joined_records"},
}


class CurriculumBuilder:
    def build_phases(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        phases = {f"phase_{index}": [] for index in CURRICULUM_PHASES}
        phases["phase_4"] = []
        for row in examples:
            intent = row.get("intent") or (row.get("query_ir") or {}).get("intent")
            placed = False
            for index, intents in CURRICULUM_PHASES.items():
                if intent in intents:
                    phases[f"phase_{index}"].append(row)
                    placed = True
                    break
            if not placed:
                phases["phase_4"].append(row)
        return phases
