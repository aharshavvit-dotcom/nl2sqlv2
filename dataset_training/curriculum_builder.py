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

    def order_examples(
        self,
        examples: list[dict[str, Any]],
        phase_order: list[str] | None = None,
        mode: str = "ordered_dataset",
        allow_ordered_dataset_fallback: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        # Guard: phased_epochs is not yet implemented
        if mode == "phased_epochs":
            if not allow_ordered_dataset_fallback:
                raise NotImplementedError(
                    "phased_epochs curriculum requested but not implemented. "
                    "Set allow_ordered_dataset_fallback=true or use mode=ordered_dataset."
                )
            mode = "ordered_dataset"  # Downgrade with honest reporting

        phases = self.build_phases(examples)
        aliases = {
            "level_1_single_table": "phase_1",
            "level_2_filter_count": "phase_1",
            "level_3_aggregation": "phase_2",
            "level_4_join": "phase_4",
            "level_5_advanced_sql": "phase_4",
        }
        requested = phase_order or ["phase_1", "phase_2", "phase_3", "phase_4"]
        canonical = []
        for name in requested:
            phase = aliases.get(name, name)
            if phase not in canonical:
                canonical.append(phase)
        canonical.extend(name for name in phases if name not in canonical)
        ordered = [row for name in canonical for row in phases.get(name, [])]
        distribution = {name: len(phases.get(name, [])) for name in phases}
        # Honest curriculum mode reporting
        distribution["_curriculum_mode"] = mode  # type: ignore[assignment]
        distribution["_phased_epochs"] = mode == "phased_epochs"  # type: ignore[assignment]
        distribution["_active"] = True  # type: ignore[assignment]
        return ordered, distribution

    def shuffle_within_buckets(
        self,
        examples: list[dict[str, Any]],
        seed: int,
    ) -> list[dict[str, Any]]:
        """Shuffle examples within their difficulty buckets, maintaining easy-to-hard phase order."""
        import random
        phases = self.build_phases(examples)
        rng = random.Random(seed)
        for name in list(phases.keys()):
            rng.shuffle(phases[name])
        ordered = []
        for name in ["phase_1", "phase_2", "phase_3", "phase_4"]:
            if name in phases:
                ordered.extend(phases[name])
        return ordered
