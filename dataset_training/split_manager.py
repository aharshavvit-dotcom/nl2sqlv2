from __future__ import annotations

import json
import hashlib
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .leakage_checker import DatasetLeakageChecker
from .utils import write_json


CANONICAL_SPLITS = (
    "train",
    "development_validation",
    "model_selection_validation",
    "frozen_semantic_test",
    "unseen_database_test",
    "controlled_execution_test",
)

REQUIRED_MANIFEST_SPLITS = (
    "train",
    "development_validation",
    "frozen_semantic_test",
    "unseen_database_test",
)

LEGACY_TO_CANONICAL_SPLIT = {
    "validation": "development_validation",
    "test": "frozen_semantic_test",
    "unseen_db_test": "unseen_database_test",
}

CANONICAL_TO_LEGACY_OUTPUT = {
    "development_validation": "validation",
    "frozen_semantic_test": "test",
    "unseen_database_test": "unseen_db_test",
}

MANIFEST_DB_ID_FIELDS = {
    "train": ("train_db_ids",),
    "development_validation": ("development_validation_db_ids", "validation_db_ids"),
    "model_selection_validation": ("model_selection_validation_db_ids",),
    "frozen_semantic_test": ("frozen_semantic_test_db_ids", "test_db_ids"),
    "unseen_database_test": ("unseen_database_test_db_ids", "unseen_db_ids"),
    "controlled_execution_test": ("controlled_execution_test_db_ids",),
}

PLACEHOLDER_DB_IDS = {f"db_{letter}" for letter in "abcdef"}


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
        force_create_new_version: bool = False,
    ):
        self.seed = seed
        self.split_version = split_version
        self.split_dir = split_dir
        self.divergence_threshold = divergence_threshold
        self.strict_mode = strict_mode
        self.force_create_new_version = force_create_new_version
        
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

    def normalize_split_name(self, split_name: str) -> str:
        return LEGACY_TO_CANONICAL_SPLIT.get(split_name, split_name)

    def output_split_name(self, split_name: str) -> str:
        return CANONICAL_TO_LEGACY_OUTPUT.get(split_name, split_name)

    def split_examples(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        return self.split_by_database(examples)

    def split_by_database(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        # 1. Load from manifest if exists and not in test environment
        in_test = "pytest" in sys.modules
        manifest_path = self.get_manifest_path()
        supported = [row for row in examples if not row.get("unsupported_reason") and row.get("query_ir") is not None]
        unsupported = [dict(row, split="unsupported") for row in examples if row not in supported]
        if not supported and not unsupported:
            return {
                "train": [],
                "validation": [],
                "test": [],
                "unseen_db_test": [],
                "unsupported": [],
            }
        if not in_test and manifest_path.exists() and not self.force_create_new_version:
            print(f"Loading split from immutable manifest: {manifest_path}")
            return self.apply_manifest_split(examples, manifest_path)
        
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
        self._reroute_non_train_eligible_rows(splits)

        # Fail early if database leakage occurs
        leakage = DatasetLeakageChecker().check_database_leakage(splits)
        if leakage["has_database_leakage"]:
            raise ValueError(f"Database leakage detected: {leakage['database_overlap']}")
        self._validate_training_eligibility(splits)

        # Save manifest if not in test environment
        if not in_test:
            self.save_manifest_split(splits, manifest_path)
            
        return splits

    def apply_manifest_split(self, examples: list[dict[str, Any]], manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest_splits = self._manifest_db_ids(manifest, manifest_path)
        self._validate_manifest(manifest, manifest_path, examples, manifest_splits)

        splits = {self.output_split_name(name): [] for name in CANONICAL_SPLITS}
        splits["unsupported"] = []

        db_to_output_split: dict[str, str] = {}
        for canonical_name, db_ids in manifest_splits.items():
            output_name = self.output_split_name(canonical_name)
            for db_id in db_ids:
                db_to_output_split[db_id] = output_name

        for row in examples:
            if row.get("unsupported_reason") or row.get("query_ir") is None:
                splits["unsupported"].append(self._with_split(row, "unsupported"))
                continue
            db_id = str(row.get("database_id") or row.get("db_id") or "")
            if db_id not in db_to_output_split:
                raise ValueError(
                    f"Database {db_id!r} is absent from split manifest {manifest_path}; "
                    "refusing to default it to train."
                )
            split_name = db_to_output_split[db_id]
            splits[split_name].append(self._with_split(row, split_name))

        self._reroute_non_train_eligible_rows(splits)
        self._validate_training_eligibility(splits)
        return splits

    def save_manifest_split(self, splits: dict[str, list[dict[str, Any]]], manifest_path: Path) -> None:
        if manifest_path.exists() and not self.force_create_new_version:
            raise FileExistsError(
                f"Split manifest already exists at {manifest_path}; split versions are immutable. "
                "Create a new split_version or set force_create_new_version=True explicitly."
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        canonical_ids: dict[str, list[str]] = {}
        for split_name, rows in splits.items():
            if split_name == "unsupported":
                continue
            canonical = self.normalize_split_name(split_name)
            canonical_ids.setdefault(canonical, [])
            canonical_ids[canonical].extend(
                str(row.get("database_id") or row.get("db_id"))
                for row in rows
                if row.get("database_id") or row.get("db_id")
            )
        canonical_ids = {
            name: sorted(set(values))
            for name, values in canonical_ids.items()
        }

        manifest = {
            "split_schema_version": "1.0",
            "split_version": self.split_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.seed,
            "dataset_hashes": self._dataset_hashes(splits),
            "source_dataset_versions": self._source_dataset_versions(splits),
            "algorithm": "group_multilabel_stratification",
            "algorithm_version": "1.0",
            "group_key": "database_id",
            "distribution_report": {},
        }
        for canonical_name, fields in MANIFEST_DB_ID_FIELDS.items():
            manifest[fields[0]] = canonical_ids.get(canonical_name, [])
            id_path = manifest_path.parent / f"{canonical_name}_ids.json"
            id_path.write_text(
                json.dumps(manifest[fields[0]], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        manifest["manifest_sha256"] = self._manifest_sha256(manifest)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved split manifest to: {manifest_path}")

    def _manifest_db_ids(self, manifest: dict[str, Any], manifest_path: Path) -> dict[str, set[str]]:
        split_ids: dict[str, set[str]] = {name: set() for name in CANONICAL_SPLITS}
        nested = manifest.get("splits")
        if isinstance(nested, dict):
            for raw_name, values in nested.items():
                canonical = self.normalize_split_name(str(raw_name))
                if canonical in split_ids and isinstance(values, list):
                    split_ids[canonical].update(str(item) for item in values)

        for canonical_name, field_names in MANIFEST_DB_ID_FIELDS.items():
            for field in field_names:
                values = manifest.get(field)
                if isinstance(values, list):
                    split_ids[canonical_name].update(str(item) for item in values)

            ids_file = manifest_path.parent / f"{canonical_name}_ids.json"
            if ids_file.exists():
                try:
                    values = json.loads(ids_file.read_text(encoding="utf-8"))
                except ValueError as exc:
                    raise ValueError(f"Invalid split id file {ids_file}: {exc}") from exc
                if not isinstance(values, list):
                    raise ValueError(f"Split id file {ids_file} must contain a JSON list.")
                split_ids[canonical_name].update(str(item) for item in values)

        return split_ids

    def _validate_manifest(
        self,
        manifest: dict[str, Any],
        manifest_path: Path,
        examples: list[dict[str, Any]],
        manifest_splits: dict[str, set[str]],
    ) -> None:
        if str(manifest.get("split_schema_version")) != "1.0":
            raise ValueError(
                f"Unsupported split manifest schema in {manifest_path}: "
                f"{manifest.get('split_schema_version')!r}"
            )

        supported_rows = [
            row for row in examples
            if not row.get("unsupported_reason") and row.get("query_ir") is not None
        ]
        example_db_ids = {
            str(row.get("database_id") or row.get("db_id"))
            for row in supported_rows
            if row.get("database_id") or row.get("db_id")
        }
        if not example_db_ids:
            raise ValueError("Cannot apply split manifest to an empty supported dataset universe.")

        db_owner: dict[str, str] = {}
        overlaps: dict[str, list[str]] = defaultdict(list)
        for split_name, db_ids in manifest_splits.items():
            for db_id in db_ids:
                if db_id in db_owner:
                    overlaps[db_id].extend([db_owner[db_id], split_name])
                db_owner[db_id] = split_name
        if overlaps:
            details = {db_id: sorted(set(names)) for db_id, names in overlaps.items()}
            raise ValueError(f"Split manifest assigns databases to multiple splits: {details}")

        manifest_db_ids = set(db_owner)
        if manifest_db_ids & PLACEHOLDER_DB_IDS:
            raise ValueError(
                f"Split manifest {manifest_path} contains reserved placeholder database IDs: "
                f"{sorted(manifest_db_ids & PLACEHOLDER_DB_IDS)}"
            )

        unknown = sorted(manifest_db_ids - example_db_ids)
        if unknown:
            raise ValueError(f"Split manifest contains unknown databases: {unknown}")
        absent = sorted(example_db_ids - manifest_db_ids)
        if absent:
            raise ValueError(
                f"Databases are absent from split manifest and cannot default to train: {absent}"
            )

        empty_required = [
            split_name for split_name in REQUIRED_MANIFEST_SPLITS
            if not manifest_splits.get(split_name)
        ]
        if empty_required:
            raise ValueError(f"Split manifest has empty required splits: {empty_required}")

        expected_hashes = manifest.get("dataset_hashes")
        if not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Split manifest is missing dataset_hashes evidence.")
        actual_hashes = self._dataset_hashes({"all": supported_rows})
        mismatched = {
            key: {"manifest": value, "actual": actual_hashes.get(key)}
            for key, value in expected_hashes.items()
            if actual_hashes.get(key) != value
        }
        if mismatched:
            raise ValueError(f"Split manifest dataset_hashes mismatch: {mismatched}")

    def _reroute_non_train_eligible_rows(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        for split_name in list(splits.keys()):
            if split_name in {"unsupported"}:
                continue
            rows = list(splits.get(split_name, []))
            for row in rows:
                if split_name != "train" or self._is_train_eligible(row):
                    continue
                target_split = self._target_split_for_row(row)
                if target_split == "train":
                    continue
                splits[split_name] = [item for item in splits[split_name] if item is not row]
                splits.setdefault(target_split, []).append(self._with_split(row, target_split))

    @staticmethod
    def _is_train_eligible(row: dict[str, Any]) -> bool:
        if row.get("eligible_for_training") is False:
            return False
        source_split = str(DatasetSplitManager._source_split(row)).lower()
        return source_split not in {"validation", "dev", "test", "unseen_db_test", "frozen", "frozen_semantic_test"}

    @staticmethod
    def _target_split_for_row(row: dict[str, Any]) -> str:
        source_split = str(DatasetSplitManager._source_split(row)).lower()
        if source_split in {"validation", "dev", "valid"}:
            return "validation"
        if source_split in {"test", "frozen", "frozen_semantic_test"}:
            return "test"
        if source_split in {"unseen_db_test", "unseen", "unseen_database_test"}:
            return "unseen_db_test"
        return "validation"

    def _validate_training_eligibility(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        violations = []
        for row in splits.get("train", []):
            source_split = self._source_split(row)
            eligible = row.get("eligible_for_training")
            if eligible is False:
                violations.append({
                    "example_id": row.get("example_id"),
                    "source_split": source_split,
                    "reason": "eligible_for_training_false",
                })
            elif source_split in {"test", "dev", "validation"}:
                violations.append({
                    "example_id": row.get("example_id"),
                    "source_split": source_split,
                    "reason": "source_split_not_train_eligible",
                })
        if violations:
            raise ValueError(
                "Source lineage forbids training assignment for records: "
                + json.dumps(violations[:20], ensure_ascii=False)
            )

    def _dataset_hashes(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rows in splits.values():
            for row in rows:
                if row.get("unsupported_reason"):
                    continue
                dataset = str(row.get("source_dataset") or row.get("dataset_name") or "unknown")
                by_dataset[dataset].append(row)
        hashes: dict[str, str] = {}
        for dataset, rows in sorted(by_dataset.items()):
            payload = []
            for row in sorted(rows, key=lambda item: str(item.get("example_id"))):
                payload.append({
                    "example_id": row.get("source_example_id") or row.get("example_id"),
                    "database_id": row.get("database_id") or row.get("db_id"),
                    "question": row.get("question"),
                    "source_sql": row.get("source_sql"),
                    "content_hash": self._content_hash_for_row(row),
                })
            hashes[dataset] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        return hashes

    @staticmethod
    def _source_dataset_versions(splits: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
        versions: dict[str, str] = {}
        for rows in splits.values():
            for row in rows:
                dataset = str(row.get("source_dataset") or row.get("dataset_name") or "unknown")
                version = str(row.get("source_dataset_version") or "unknown")
                versions.setdefault(dataset, version)
        return versions

    @staticmethod
    def _manifest_sha256(manifest: dict[str, Any]) -> str:
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _content_hash_for_row(row: dict[str, Any]) -> str:
        if row.get("content_hash"):
            return str(row["content_hash"])
        source_dataset = row.get("source_dataset") or row.get("dataset_name") or row.get("dataset") or "unknown"
        source_example_id = row.get("source_example_id") or row.get("example_id")
        database_id = row.get("database_id") or row.get("db_id")
        return hashlib.sha256(
            json.dumps(
                {
                    "source_dataset": source_dataset,
                    "source_example_id": source_example_id,
                    "database_id": database_id,
                    "question": row.get("question"),
                    "source_sql": row.get("source_sql") or row.get("gold_sql"),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _source_split(row: dict[str, Any]) -> str:
        metadata = row.get("metadata") or {}
        return str(
            row.get("source_split")
            or metadata.get("source_split")
            or metadata.get("split")
            or row.get("split")
            or ""
        ).lower()

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
        metadata = dict(updated.get("metadata") or {})
        source_split = (
            updated.get("source_split")
            or metadata.get("source_split")
            or metadata.get("split")
            or updated.get("split")
            or split
        )
        source_dataset = updated.get("source_dataset") or updated.get("dataset_name") or updated.get("dataset") or "unknown"
        source_example_id = updated.get("source_example_id") or updated.get("example_id")
        database_id = updated.get("database_id") or updated.get("db_id")
        content_hash = DatasetSplitManager._content_hash_for_row(updated)

        updated["source_dataset"] = source_dataset
        updated["source_dataset_version"] = updated.get("source_dataset_version") or metadata.get("source_dataset_version") or "unknown"
        updated["source_split"] = source_split
        updated["source_example_id"] = source_example_id
        updated["database_id"] = database_id
        updated["internal_split"] = split
        updated["eligible_for_training"] = bool(
            updated.get("eligible_for_training", str(source_split).lower() == "train")
        )
        updated["content_hash"] = content_hash
        updated["split"] = split
        updated["metadata"] = {
            **metadata,
            "source_dataset": source_dataset,
            "source_dataset_version": updated["source_dataset_version"],
            "source_split": source_split,
            "source_example_id": source_example_id,
            "internal_split": split,
            "content_hash": content_hash,
        }
        if isinstance(updated.get("query_ir"), dict):
            updated["query_ir"] = dict(updated["query_ir"])
            updated["query_ir"].setdefault("metadata", {})["split"] = split
            updated["query_ir"].setdefault("metadata", {})["internal_split"] = split
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
