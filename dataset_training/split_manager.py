from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .leakage_checker import DatasetLeakageChecker
from .utils import write_json


class DatasetSplitManager:
    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        model_selection_ratio: float = 0.0,
        test_ratio: float = 0.1,
        unseen_db_test_ratio: float = 0.15,
        split_version: str = "semantic_v1",
        split_dir: str | Path | None = None,
        divergence_threshold: float = 0.25,
        strict_mode: bool = False,
    ):
        self.seed = seed
        self.split_version = split_version
        self.split_dir = split_dir
        self.divergence_threshold = divergence_threshold
        self.strict_mode = strict_mode
        
        self.raw_train_ratio = train_ratio
        self.raw_validation_ratio = validation_ratio
        self.raw_model_selection_ratio = model_selection_ratio
        self.raw_test_ratio = test_ratio
        
        total = train_ratio + validation_ratio + model_selection_ratio + test_ratio
        self.train_ratio = train_ratio / total
        self.validation_ratio = validation_ratio / total
        self.model_selection_ratio = model_selection_ratio / total
        self.test_ratio = test_ratio / total
        self.unseen_db_test_ratio = max(0.0, min(unseen_db_test_ratio, 0.8))

    def get_manifest_path(self) -> Path:
        root = Path(__file__).resolve().parents[1]
        if self.split_dir:
            return Path(self.split_dir) / self.split_version / "split_manifest.json"
        return root / "data" / "splits" / self.split_version / "split_manifest.json"

    def split_examples(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        return self.split_by_database(examples)

    def split_by_database(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        # 1. Load from manifest if exists and not in test environment
        in_test = "pytest" in sys.modules
        manifest_path = self.get_manifest_path()
        if not in_test and manifest_path.exists():
            print(f"Loading split from immutable manifest: {manifest_path}")
            return self.apply_manifest_split(examples, manifest_path)
            
        supported = [row for row in examples if not row.get("unsupported_reason") and row.get("query_ir") is not None]
        unsupported = [dict(row, split="unsupported") for row in examples if row not in supported]
        
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in supported:
            grouped[str(row.get("db_id") or "__unknown_db__")].append(row)

        db_ids = sorted(grouped)
        
        # Unseen DB partition isolation
        random.Random(self.seed).shuffle(db_ids)
        unseen_count = 0
        if len(db_ids) >= 2 and self.unseen_db_test_ratio > 0:
            unseen_count = max(1, int(round(len(db_ids) * self.unseen_db_test_ratio)))
            unseen_count = min(unseen_count, len(db_ids) - 1)
        unseen_dbs = set(db_ids[:unseen_count])
        regular_dbs = db_ids[unseen_count:]

        # Group-Size-Aware Multilabel Stratification
        splits_info = {
            "train": self.train_ratio,
            "validation": self.validation_ratio,
        }
        if self.raw_model_selection_ratio > 0:
            splits_info["model_selection_validation"] = self.model_selection_ratio
        if self.raw_test_ratio > 0:
            splits_info["test"] = self.test_ratio
            
        allocated_dbs = self._stratify_groups(regular_dbs, grouped, splits_info)
        
        train_dbs = {db_id for db_id, s in allocated_dbs.items() if s == "train"}
        validation_dbs = {db_id for db_id, s in allocated_dbs.items() if s == "validation"}
        model_selection_dbs = {db_id for db_id, s in allocated_dbs.items() if s == "model_selection_validation"}
        test_dbs = {db_id for db_id, s in allocated_dbs.items() if s == "test"}

        splits = {
            "train": [row for db_id in regular_dbs if db_id in train_dbs for row in grouped[db_id]],
            "validation": [row for db_id in regular_dbs if db_id in validation_dbs for row in grouped[db_id]],
        }
        if self.raw_model_selection_ratio > 0:
            splits["model_selection_validation"] = [row for db_id in regular_dbs if db_id in model_selection_dbs for row in grouped[db_id]]
        if self.raw_test_ratio > 0:
            splits["test"] = [row for db_id in regular_dbs if db_id in test_dbs for row in grouped[db_id]]
            
        splits["unseen_db_test"] = [row for db_id in unseen_dbs for row in grouped[db_id]]
        splits["unsupported"] = unsupported

        for name, rows in splits.items():
            splits[name] = [self._with_split(row, name) for row in rows]

        # Fail early if database leakage occurs
        leakage = DatasetLeakageChecker().check_database_leakage(splits)
        if leakage["has_database_leakage"]:
            raise ValueError(f"Database leakage detected: {leakage['database_overlap']}")

        # Save manifest if not in test environment
        if not in_test:
            self.save_manifest_split(splits, manifest_path)
            
        return splits

    def apply_manifest_split(self, examples: list[dict[str, Any]], manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        train_db_ids = set(manifest.get("train_db_ids", []))
        validation_db_ids = set(manifest.get("validation_db_ids", []))
        model_selection_db_ids = set(manifest.get("model_selection_db_ids", []))
        test_db_ids = set(manifest.get("test_db_ids", []))
        unseen_db_ids = set(manifest.get("unseen_db_ids", []))
        
        splits = {
            "train": [],
            "validation": [],
        }
        if self.raw_model_selection_ratio > 0 or model_selection_db_ids:
            splits["model_selection_validation"] = []
        if self.raw_test_ratio > 0 or test_db_ids:
            splits["test"] = []
        splits["unseen_db_test"] = []
        splits["unsupported"] = []
        
        for row in examples:
            if row.get("unsupported_reason") or row.get("query_ir") is None:
                splits["unsupported"].append(self._with_split(row, "unsupported"))
                continue
            db_id = str(row.get("db_id"))
            if db_id in train_db_ids:
                splits["train"].append(self._with_split(row, "train"))
            elif db_id in validation_db_ids:
                splits["validation"].append(self._with_split(row, "validation"))
            elif db_id in model_selection_db_ids and "model_selection_validation" in splits:
                splits["model_selection_validation"].append(self._with_split(row, "model_selection_validation"))
            elif db_id in test_db_ids and "test" in splits:
                splits["test"].append(self._with_split(row, "test"))
            elif db_id in unseen_db_ids:
                splits["unseen_db_test"].append(self._with_split(row, "unseen_db_test"))
            else:
                splits["train"].append(self._with_split(row, "train"))
                
        return splits

    def save_manifest_split(self, splits: dict[str, list[dict[str, Any]]], manifest_path: Path) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_db_ids = sorted({str(row.get("db_id")) for row in splits["train"] if row.get("db_id")})
        validation_db_ids = sorted({str(row.get("db_id")) for row in splits["validation"] if row.get("db_id")})
        model_selection_db_ids = sorted({str(row.get("db_id")) for row in splits.get("model_selection_validation", []) if row.get("db_id")})
        test_db_ids = sorted({str(row.get("db_id")) for row in splits.get("test", []) if row.get("db_id")})
        unseen_db_ids = sorted({str(row.get("db_id")) for row in splits["unseen_db_test"] if row.get("db_id")})
        
        manifest = {
            "split_schema_version": "1.0",
            "split_version": self.split_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.seed,
            "dataset_hashes": {},
            "algorithm": "group_multilabel_stratification",
            "algorithm_version": "1.0",
            "group_key": "database_id",
            "train_db_ids": train_db_ids,
            "validation_db_ids": validation_db_ids,
            "model_selection_db_ids": model_selection_db_ids,
            "test_db_ids": test_db_ids,
            "unseen_db_ids": unseen_db_ids,
        }
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved split manifest to: {manifest_path}")

    def split_by_dataset_and_database(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in examples:
            by_dataset[str(row.get("dataset_name") or "unknown")].append(row)
            
        merged = {
            "train": [],
            "validation": [],
        }
        if self.raw_model_selection_ratio > 0:
            merged["model_selection_validation"] = []
        if self.raw_test_ratio > 0:
            merged["test"] = []
        merged["unseen_db_test"] = []
        merged["unsupported"] = []
        
        for dataset, rows in sorted(by_dataset.items()):
            dataset_splits = self.split_by_database(rows)
            for split_name, split_rows in dataset_splits.items():
                merged[split_name].extend(split_rows)
        return merged

    def save_split_report(self, splits: dict[str, list[dict[str, Any]]], output_path: str) -> None:
        split_names = list(splits.keys())
        report = {
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "database_counts": {name: len({row.get("db_id") for row in rows}) for name, rows in splits.items()},
            "databases": {name: sorted({str(row.get("db_id")) for row in rows if row.get("db_id")}) for name, rows in splits.items()},
            **{
                name: {
                    "by_dataset": _distribution(rows, lambda row: row.get("dataset_name") or "unknown"),
                    "by_intent": _distribution(rows, lambda row: row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"),
                    "by_complexity": _distribution(rows, lambda row: row.get("complexity") or "unknown"),
                    "by_join_count": _distribution(rows, lambda row: len((row.get("query_ir") or {}).get("joins") or [])),
                    "by_aggregation_type": _distribution(rows, _aggregation_type),
                }
                for name, rows in splits.items()
            },
        }
        write_json(Path(output_path), report)
        target = Path(output_path)
        lines = ["# Split Distribution Report", ""]
        for name in split_names:
            lines.extend([f"## {name}", "", f"- examples: {len(splits.get(name, []))}", f"- databases: {report['database_counts'].get(name, 0)}", f"- intents: {report.get(name, {}).get('by_intent', {})}", f"- complexity: {report.get(name, {}).get('by_complexity', {})}", ""])
        target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
        updated = dict(row)
        updated["split"] = split
        if isinstance(updated.get("query_ir"), dict):
            updated["query_ir"] = dict(updated["query_ir"])
            updated["query_ir"].setdefault("metadata", {})["split"] = split
        return updated

    def _stratify_groups(
        self,
        db_ids: list[str],
        grouped: dict[str, list[dict[str, Any]]],
        splits_info: dict[str, float],
    ) -> dict[str, str]:
        db_profiles = {}
        label_total_counts = defaultdict(int)
        for db_id in db_ids:
            profile = self._get_db_label_profile(grouped[db_id])
            db_profiles[db_id] = profile
            for label, count in profile.items():
                label_total_counts[label] += count
                
        split_names = list(splits_info.keys())
        total_examples = sum(len(grouped[db_id]) for db_id in db_ids)
        split_targets = {name: ratio * total_examples for name, ratio in splits_info.items()}
        split_allocated = {name: 0 for name in split_names}
        
        label_targets = {}
        for label, total_count in label_total_counts.items():
            label_targets[label] = {name: ratio * total_count for name, ratio in splits_info.items()}
        label_allocated = {label: {name: 0 for name in split_names} for label in label_total_counts}
        
        unallocated_dbs = set(db_ids)
        allocated_split = {}
        
        sorted_labels = sorted(label_total_counts.keys(), key=lambda l: label_total_counts[l])
        
        for label in sorted_labels:
            containing_dbs = [db_id for db_id in unallocated_dbs if label in db_profiles[db_id]]
            if not containing_dbs:
                continue
            containing_dbs.sort(key=lambda db_id: len(grouped[db_id]), reverse=True)
            
            for db_id in containing_dbs:
                if db_id not in unallocated_dbs:
                    continue
                db_size = len(grouped[db_id])
                db_profile = db_profiles[db_id]
                
                best_split = None
                best_score = -1.0
                for name in split_names:
                    target_l = label_targets[label][name]
                    alloc_l = label_allocated[label][name]
                    deficit_l = target_l - alloc_l
                    
                    target_size = split_targets[name]
                    alloc_size = split_allocated[name]
                    deficit_size = target_size - alloc_size
                    
                    score = (deficit_l / max(target_l, 1.0)) * 0.7 + (deficit_size / max(target_size, 1.0)) * 0.3
                    if best_split is None or score > best_score:
                        best_score = score
                        best_split = name
                        
                allocated_split[db_id] = best_split
                unallocated_dbs.remove(db_id)
                split_allocated[best_split] += db_size
                for lbl, count in db_profile.items():
                    label_allocated[lbl][best_split] += count
                    
        while unallocated_dbs:
            db_id = max(unallocated_dbs, key=lambda db_id: len(grouped[db_id]))
            db_size = len(grouped[db_id])
            db_profile = db_profiles[db_id]
            best_split = max(split_names, key=lambda name: (split_targets[name] - split_allocated[name]) / max(split_targets[name], 1.0))
            allocated_split[db_id] = best_split
            unallocated_dbs.remove(db_id)
            split_allocated[best_split] += db_size
            for lbl, count in db_profile.items():
                label_allocated[lbl][best_split] += count
                
        divergences = self._calculate_divergence(label_allocated, label_total_counts, splits_info)
        print(f"Stratified Split Divergence Report: {json.dumps(divergences, indent=2)}")
        
        for s_name, s_divs in divergences.items():
            for feat_name, l1_div in s_divs.items():
                if l1_div > self.divergence_threshold:
                    msg = f"Divergence threshold exceeded on split {s_name} feature {feat_name}: actual={l1_div:.4f}, limit={self.divergence_threshold:.4f}"
                    if self.strict_mode:
                        raise ValueError(msg)
                    else:
                        print(f"Warning: {msg}")
                        
        return allocated_split

    def _get_db_label_profile(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        profile = defaultdict(int)
        for row in rows:
            dataset = row.get("dataset_name") or "unknown"
            profile[f"dataset:{dataset}"] += 1
            
            intent = row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"
            profile[f"intent:{intent}"] += 1
            
            complexity = row.get("complexity") or "unknown"
            profile[f"complexity:{complexity}"] += 1
            
            joins = len((row.get("query_ir") or {}).get("joins") or [])
            j_bucket = "0" if joins == 0 else ("1" if joins == 1 else ("2" if joins == 2 else "3+"))
            profile[f"joins:{j_bucket}"] += 1
            
            filters = len((row.get("query_ir") or {}).get("filters") or [])
            f_bucket = "0" if filters == 0 else ("1" if filters == 1 else ("2" if filters == 2 else "3+"))
            profile[f"filters:{f_bucket}"] += 1
            
            schema = row.get("schema") or {}
            tables = schema.get("tables") or {}
            size = len(tables) if isinstance(tables, dict) else 0
            s_bucket = "small" if size <= 5 else ("medium" if size <= 15 else "large")
            profile[f"schema_size:{s_bucket}"] += 1
            
            agg = _aggregation_type(row)
            profile[f"aggregation:{agg}"] += 1
            
            df = "true" if bool((row.get("query_ir") or {}).get("date_filters")) else "false"
            profile[f"date_filter:{df}"] += 1
            
            gb = "true" if bool((row.get("query_ir") or {}).get("group_by")) else "false"
            profile[f"group_by:{gb}"] += 1
            
            ob = "true" if bool((row.get("query_ir") or {}).get("order_by")) else "false"
            profile[f"order_by:{ob}"] += 1
            
            lim = "true" if bool((row.get("query_ir") or {}).get("limit")) else "false"
            profile[f"limit:{lim}"] += 1
            
        return dict(profile)

    def _calculate_divergence(
        self,
        label_allocated: dict[str, dict[str, int]],
        label_total_counts: dict[str, int],
        splits_info: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        features = ["dataset", "intent", "complexity", "joins", "filters", "schema_size", "aggregation", "date_filter", "group_by", "order_by", "limit"]
        divergences = {}
        for name in splits_info:
            divergences[name] = {}
            for feat in features:
                feat_total = sum(count for lbl, count in label_total_counts.items() if lbl.startswith(feat + ":"))
                if feat_total == 0:
                    continue
                allocated_feat_total = sum(label_allocated[l][name] for l in label_total_counts if l.startswith(feat + ":"))
                l1_distance = 0.0
                for lbl, count in label_total_counts.items():
                    if lbl.startswith(feat + ":"):
                        target_prop = count / feat_total
                        allocated_prop = label_allocated[lbl][name] / max(allocated_feat_total, 1.0)
                        l1_distance += abs(allocated_prop - target_prop)
                divergences[name][feat] = l1_distance / 2.0
        return divergences


def _distribution(rows: list[dict[str, Any]], key: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(key(row))] += 1
    return dict(sorted(counts.items()))


def _aggregation_type(row: dict[str, Any]) -> str:
    metrics = (row.get("query_ir") or {}).get("metrics") or []
    aggregations = sorted({str(item.get("aggregation") or "none") for item in metrics if isinstance(item, dict)})
    return "+".join(aggregations) if aggregations else "none"
