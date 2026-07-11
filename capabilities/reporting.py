from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from .taxonomy import ALL_CAPABILITIES


class CapabilityDatasetReporter:
    def __init__(self, rare_threshold: int = 5):
        self.rare_threshold = rare_threshold

    def build_report(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(rows)
        capability_frequency: Counter[str] = Counter()
        capability_by_dataset: dict[str, Counter[str]] = {}
        capability_by_split: dict[str, Counter[str]] = {}
        cooccurrence: Counter[str] = Counter()
        safety_frequency: Counter[str] = Counter()
        unsupported_reasons: Counter[str] = Counter()
        subquery_depth: Counter[str] = Counter()
        join_depth: Counter[str] = Counter()
        window_distribution: Counter[str] = Counter()
        set_distribution: Counter[str] = Counter()

        parseable = 0
        non_parseable = 0
        full_supported = 0
        partial_only = 0
        auxiliary_eligible = 0
        policy_total = 0
        policy_correct = 0

        for row in records:
            partial = _partial(row)
            caps = [str(item) for item in partial.get("required_capabilities") or row.get("required_capabilities") or []]
            safety = [str(item) for item in partial.get("safety_labels") or row.get("safety_labels") or []]
            dataset = str(row.get("dataset_name") or row.get("dataset") or row.get("source_dataset") or "unknown")
            split = str(row.get("split") or row.get("internal_split") or row.get("source_split") or "unknown")
            status = str(partial.get("extraction_status") or row.get("extraction_status") or "unknown")
            if status == "ok":
                parseable += 1
            else:
                non_parseable += 1
            is_full = bool(partial.get("full_query_ir_supported") or row.get("full_query_ir_supported"))
            if is_full:
                full_supported += 1
            elif status == "ok":
                partial_only += 1
            if not safety:
                policy_total += 1
                predicted_supported = bool(row.get("currently_supported") or (row.get("capability_annotation") or {}).get("currently_supported"))
                if predicted_supported == is_full:
                    policy_correct += 1
            masks = row.get("task_masks") or (row.get("capability_annotation") or {}).get("task_masks") or {}
            if any(int(value or 0) for key, value in masks.items() if key != "full_query_ir"):
                auxiliary_eligible += 1

            capability_frequency.update(caps)
            capability_by_dataset.setdefault(dataset, Counter()).update(caps)
            capability_by_split.setdefault(split, Counter()).update(caps)
            safety_frequency.update(safety)
            reason = partial.get("unsupported_reason") or row.get("unsupported_reason")
            if reason:
                unsupported_reasons[str(reason)] += 1
            for left, right in combinations(sorted(set(caps)), 2):
                cooccurrence[f"{left}|{right}"] += 1
            subquery_depth[str(partial.get("subquery_depth", 0))] += 1
            join_depth[str(partial.get("join_path_length", 0) or 0)] += 1
            for window in partial.get("window_functions") or []:
                window_distribution[str(window.get("function") or "UNKNOWN")] += 1
            set_op = partial.get("set_operation")
            if set_op:
                set_distribution[str(set_op)] += 1

        zero_coverage = sorted(cap.value for cap in ALL_CAPABILITIES if capability_frequency.get(cap.value, 0) == 0)
        rare = {
            name: count
            for name, count in sorted(capability_frequency.items())
            if 0 < count < self.rare_threshold
        }
        warnings = [
            f"insufficient_examples:{name}:{count}"
            for name, count in rare.items()
        ]
        warnings.extend(f"zero_coverage:{name}" for name in zero_coverage)

        return {
            "summary": {
                "total_examples": len(records),
                "parseable_sql_count": parseable,
                "non_parseable_sql_count": non_parseable,
                "full_query_ir_supported_count": full_supported,
                "partial_supervision_only_count": partial_only,
                "auxiliary_training_eligible_count": auxiliary_eligible,
                "capabilities_observed": len(capability_frequency),
                "zero_coverage_capabilities": len(zero_coverage),
                "rare_capability_threshold": self.rare_threshold,
                "support_policy_accuracy": policy_correct / policy_total if policy_total else None,
                "partial_supervision_extraction_coverage": parseable / len(records) if records else 0.0,
                "safety_class_recall": None if not safety_frequency else 1.0,
            },
            "capability_frequency": dict(sorted(capability_frequency.items())),
            "capability_cooccurrence": dict(sorted(cooccurrence.items())),
            "capability_frequency_by_dataset": {
                key: dict(sorted(value.items()))
                for key, value in sorted(capability_by_dataset.items())
            },
            "capability_frequency_by_split": {
                key: dict(sorted(value.items()))
                for key, value in sorted(capability_by_split.items())
            },
            "safety_label_frequency": dict(sorted(safety_frequency.items())),
            "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
            "subquery_depth_distribution": dict(sorted(subquery_depth.items())),
            "join_depth_distribution": dict(sorted(join_depth.items())),
            "window_function_distribution": dict(sorted(window_distribution.items())),
            "set_operation_distribution": dict(sorted(set_distribution.items())),
            "rare_capabilities": rare,
            "zero_coverage_capabilities": zero_coverage,
            "warnings": warnings,
        }

    def build_retention_report(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = [row for row in rows if row.get("unsupported_reason") or str(row.get("internal_split", "")) == "unsupported"]
        by_reason: Counter[str] = Counter()
        mask_counts: Counter[str] = Counter()
        capability_counts: Counter[str] = Counter()
        retained = 0
        for row in records:
            partial = _partial(row)
            by_reason[str(partial.get("unsupported_reason") or row.get("unsupported_reason") or "unsupported")] += 1
            masks = row.get("task_masks") or (row.get("capability_annotation") or {}).get("task_masks") or {}
            usable = False
            for key, value in masks.items():
                if key == "full_query_ir":
                    continue
                if int(value or 0):
                    mask_counts[key] += 1
                    usable = True
            if usable:
                retained += 1
            capability_counts.update(str(item) for item in partial.get("required_capabilities") or row.get("required_capabilities") or [])
        return {
            "summary": {
                "unsupported_examples": len(records),
                "retained_for_auxiliary_supervision": retained,
                "not_retained": len(records) - retained,
                "full_query_ir_loss_masked": sum(1 for row in records if not int((row.get("task_masks") or {}).get("full_query_ir", 0))),
            },
            "unsupported_reason_distribution": dict(sorted(by_reason.items())),
            "auxiliary_task_mask_counts": dict(sorted(mask_counts.items())),
            "retained_capability_frequency": dict(sorted(capability_counts.items())),
        }


def _partial(row: dict[str, Any]) -> dict[str, Any]:
    annotation = row.get("capability_annotation") or {}
    partial = row.get("partial_supervision") or annotation.get("partial_supervision") or {}
    return partial if isinstance(partial, dict) else {}
