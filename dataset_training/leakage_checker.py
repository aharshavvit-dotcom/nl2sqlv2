from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from typing import Any

from .utils import normalize_text


class DatasetLeakageError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("Dataset leakage checks failed")
        self.report = report


def get_structural_ast_signature(sql: str, dialect: str = "sqlite") -> str:
    import sqlglot
    from sqlglot import exp
    try:
        ast = sqlglot.parse_one(sql, read=dialect)
        for node in ast.find_all(exp.Literal):
            node.replace(exp.Literal.string("?"))
        return ast.sql(dialect=dialect, pretty=False).lower()
    except Exception:
        val = re.sub(r"'\w+'|\b\d+\b", "?", sql).lower()
        return " ".join(val.split())


def get_semantic_ast_signature(sql: str, dialect: str = "sqlite") -> str:
    import sqlglot
    try:
        ast = sqlglot.parse_one(sql, read=dialect)
        return ast.sql(dialect=dialect, pretty=False).lower()
    except Exception:
        return " ".join(sql.lower().split())


class DatasetLeakageChecker:
    def check_database_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        dbs = {
            name: {
                str(row.get("database_id") or row.get("db_id"))
                for row in rows
                if row.get("database_id") or row.get("db_id")
            }
            for name, rows in splits.items()
            if name != "unsupported"
        }
        overlap: dict[str, list[str]] = {}
        for left, right in combinations(sorted(dbs), 2):
            shared = sorted(dbs[left] & dbs[right])
            if shared:
                overlap[f"{left}__{right}"] = shared
        train_like = (
            dbs.get("train", set())
            | dbs.get("validation", set())
            | dbs.get("development_validation", set())
            | dbs.get("model_selection_validation", set())
        )
        unseen_like = dbs.get("unseen_db_test", set()) | dbs.get("unseen_database_test", set())
        train_unseen = sorted(train_like & unseen_like)
        return {
            "has_database_leakage": bool(overlap),
            "database_overlap": overlap,
            "train_unseen_overlap": train_unseen,
        }

    def check_question_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        train_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in splits.get("train", []):
            value = normalize_text(row.get("question"))
            if value:
                train_values[value].append(row)

        violations: list[dict[str, Any]] = []
        generic_overlaps: list[dict[str, Any]] = []
        for split_name in _comparison_split_names(splits):
            for row in splits.get(split_name, []):
                value = normalize_text(row.get("question"))
                if not value or value not in train_values:
                    continue
                for train_row in train_values[value]:
                    item = {
                        "normalized_question": value,
                        "train_example_id": train_row.get("example_id"),
                        "other_example_id": row.get("example_id"),
                        "other_split": split_name,
                        "train_database_id": train_row.get("database_id") or train_row.get("db_id"),
                        "other_database_id": row.get("database_id") or row.get("db_id"),
                    }
                    if self._is_generic_template_overlap(value, train_row, row):
                        generic_overlaps.append({**item, "blocking": False})
                    else:
                        violations.append({**item, "blocking": True})
        return {
            "has_question_leakage": bool(violations),
            "question_overlap_count": len(violations),
            "question_overlap": sorted({item["normalized_question"] for item in violations}),
            "question_leakage_violations": violations,
            "generic_template_overlap_count": len(generic_overlaps),
            "generic_template_overlaps": generic_overlaps,
        }

    def check_sql_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        train_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in splits.get("train", []):
            value = normalize_text(row.get("source_sql"))
            if value:
                train_values[value].append(row)

        violations: list[dict[str, Any]] = []
        generic_overlaps: list[dict[str, Any]] = []
        for split_name in _comparison_split_names(splits):
            for row in splits.get(split_name, []):
                value = normalize_text(row.get("source_sql"))
                if not value or value not in train_values:
                    continue
                for train_row in train_values[value]:
                    item = {
                        "normalized_sql": value,
                        "source_sql": row.get("source_sql"),
                        "train_example_id": train_row.get("example_id"),
                        "other_example_id": row.get("example_id"),
                        "other_split": split_name,
                        "train_database_id": train_row.get("database_id") or train_row.get("db_id"),
                        "other_database_id": row.get("database_id") or row.get("db_id"),
                    }
                    if self._is_generic_sql_overlap(train_row, row):
                        generic_overlaps.append({**item, "blocking": False})
                    else:
                        violations.append({**item, "blocking": True})
        return {
            "has_sql_leakage": bool(violations),
            "sql_overlap_count": len(violations),
            "source_sql_overlap": sorted({item["normalized_sql"] for item in violations}),
            "sql_leakage_violations": violations,
            "generic_sql_overlap_count": len(generic_overlaps),
            "generic_sql_overlaps": generic_overlaps,
        }

    def check_query_ir_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        train_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in splits.get("train", []):
            signature = self._canonical_query_ir_signature(row.get("query_ir"))
            if signature:
                train_values[signature].append(row)

        violations: list[dict[str, Any]] = []
        for split_name in _comparison_split_names(splits):
            for row in splits.get(split_name, []):
                signature = self._canonical_query_ir_signature(row.get("query_ir"))
                if not signature or signature not in train_values:
                    continue
                for train_row in train_values[signature]:
                    violations.append({
                        "query_ir_hash": signature,
                        "train_example_id": train_row.get("example_id"),
                        "other_example_id": row.get("example_id"),
                        "other_split": split_name,
                        "leakage_type": "canonical_query_ir_leakage",
                        "blocking": True,
                    })
                    break
        return {
            "has_query_ir_leakage": bool(violations),
            "query_ir_overlap_count": len(violations),
            "query_ir_violations": violations,
        }

    def check_parent_child_lineage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        id_to_split = {}
        parent_map = {}
        for split_name, rows in splits.items():
            for row in rows:
                if "example_id" in row:
                    id_to_split[str(row["example_id"])] = split_name
                    parent_id = (
                        (row.get("metadata") or {}).get("original_example_id")
                        or (row.get("metadata") or {}).get("augmentation_parent_id")
                        or row.get("parent_example_id")
                    )
                    if parent_id:
                        parent_map[str(row["example_id"])] = str(parent_id)
                    
        violations = []
        for split_name, rows in splits.items():
            for row in rows:
                example_id = str(row.get("example_id"))
                lineage = []
                parent_id = parent_map.get(example_id)
                visited = set()
                while parent_id and parent_id not in visited:
                    visited.add(parent_id)
                    lineage.append(parent_id)
                    if parent_id in id_to_split:
                        parent_split = id_to_split[parent_id]
                        if parent_split != split_name:
                            violations.append({
                                "example_id": row.get("example_id"),
                                "parent_id": parent_id,
                                "lineage": lineage,
                                "example_split": split_name,
                                "parent_split": parent_split
                            })
                            break
                    parent_id = parent_map.get(parent_id)
        return {
            "has_parent_child_violations": bool(violations),
            "parent_child_violations": violations
        }

    def check_schema_family_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        db_fingerprints = {}
        for split_name, rows in splits.items():
            for row in rows:
                db_id = row.get("database_id") or row.get("db_id")
                if not db_id or db_id in db_fingerprints:
                    continue
                schema = row.get("schema") or {}
                tables = schema.get("tables") or {}
                if not tables:
                    continue
                sig_parts = []
                for t_name in sorted(tables.keys()):
                    cols = tables[t_name]
                    col_sig = []
                    if isinstance(cols, dict):
                        col_sig = sorted(f"{c}:{t}" for c, t in cols.items())
                    elif isinstance(cols, list):
                        col_sig = sorted(str(c) for c in cols)
                    sig_parts.append(f"{t_name}({','.join(col_sig)})")
                fingerprint = hashlib.sha256(";".join(sig_parts).encode("utf-8")).hexdigest()
                db_fingerprints[db_id] = fingerprint
                
        fingerprint_to_dbs = defaultdict(list)
        for db_id, fp in db_fingerprints.items():
            fingerprint_to_dbs[fp].append(db_id)
            
        split_to_fingerprints = defaultdict(set)
        for split_name, rows in splits.items():
            for row in rows:
                db_id = row.get("database_id") or row.get("db_id")
                if db_id and db_id in db_fingerprints:
                    split_to_fingerprints[split_name].add(db_fingerprints[db_id])
                    
        train_val_fps = (
            split_to_fingerprints.get("train", set())
            | split_to_fingerprints.get("validation", set())
            | split_to_fingerprints.get("development_validation", set())
            | split_to_fingerprints.get("model_selection_validation", set())
        )
        unseen_fps = split_to_fingerprints.get("unseen_db_test", set()) | split_to_fingerprints.get("unseen_database_test", set())
        
        shared_fps = train_val_fps & unseen_fps
        overlap_details = []
        for fp in shared_fps:
            overlap_details.append({
                "fingerprint": fp,
                "databases": fingerprint_to_dbs[fp]
            })
            
        return {
            "has_schema_family_leakage": bool(shared_fps),
            "schema_family_overlaps": overlap_details
        }

    def check_sql_ast_leakage(self, splits: dict[str, list[dict[str, Any]]], dialect: str = "sqlite") -> dict[str, Any]:
        train_semantic_sigs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in splits.get("train", []):
            sql = row.get("source_sql")
            if not sql:
                continue
            semantic_sig = get_semantic_ast_signature(sql, dialect)
            train_semantic_sigs[semantic_sig].append(row)
            
        violations = []
        generic_overlaps = []
        for split_name in _comparison_split_names(splits):
            for row in splits.get(split_name, []):
                sql = row.get("source_sql")
                if not sql:
                    continue
                semantic_sig = get_semantic_ast_signature(sql, dialect)
                if semantic_sig not in train_semantic_sigs:
                    continue
                for train_row in train_semantic_sigs[semantic_sig]:
                    if self._is_generic_sql_overlap(train_row, row):
                        generic_overlaps.append({
                            "query_signature": semantic_sig,
                            "train_example_id": train_row.get("example_id"),
                            "other_example_id": row.get("example_id"),
                            "other_split": split_name,
                            "blocking": False,
                        })
                        continue
                    violations.append({
                        "example_id": row.get("example_id"),
                        "train_example_id": train_row.get("example_id"),
                        "question": row.get("question"),
                        "sql": sql,
                        "split": split_name,
                        "leakage_type": "exact_sql_ast_leakage"
                    })
                    break
        return {
            "has_sql_ast_leakage": bool(violations),
            "sql_ast_violations": violations,
            "generic_sql_ast_overlap_count": len(generic_overlaps),
            "generic_sql_ast_overlaps": generic_overlaps,
        }

    def check_near_duplicate_leakage(self, splits: dict[str, list[dict[str, Any]]], token_threshold: float = 0.85, char_threshold: float = 0.85) -> dict[str, Any]:
        train_by_intent = defaultdict(list)
        for row in splits.get("train", []):
            intent = row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"
            q = row.get("question")
            if q:
                train_by_intent[intent].append((row.get("example_id"), q, set(q.lower().split()), row))
                
        violations = []
        for split_name in _comparison_split_names(splits):
            for row in splits.get(split_name, []):
                intent = row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"
                q = row.get("question")
                if not q:
                    continue
                tokens = set(q.lower().split())
                for train_id, train_q, train_tokens, train_row in train_by_intent[intent]:
                    tok_sim = len(tokens & train_tokens) / max(len(tokens | train_tokens), 1)
                    if tok_sim >= token_threshold:
                        if self._is_generic_template_overlap(normalize_text(q), train_row, row):
                            continue
                        char1 = set(q.lower()[i:i+3] for i in range(len(q)-2))
                        char2 = set(train_q.lower()[i:i+3] for i in range(len(train_q)-2))
                        char_sim = len(char1 & char2) / max(len(char1 | char2), 1)
                        if char_sim >= char_threshold:
                            violations.append({
                                "left_example_id": train_id,
                                "right_example_id": row.get("example_id"),
                                "left_question": train_q,
                                "right_question": q,
                                "right_split": split_name,
                                "token_similarity": tok_sim,
                                "character_similarity": char_sim,
                                "leakage_type": "near_duplicate_leakage",
                                "blocking": True
                            })
                            break
        return {
            "has_near_duplicate_leakage": bool(violations),
            "near_duplicate_violations": violations
        }

    def run_all_checks(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        database = self.check_database_leakage(splits)
        question = self.check_question_leakage(splits)
        sql = self.check_sql_leakage(splits)
        query_ir = self.check_query_ir_leakage(splits)
        parent_child = self.check_parent_child_lineage(splits)
        schema_family = self.check_schema_family_leakage(splits)
        sql_ast = self.check_sql_ast_leakage(splits)
        near_duplicate = self.check_near_duplicate_leakage(splits)
        
        result = {
            **database,
            **question,
            **sql,
            **query_ir,
            **parent_child,
            **schema_family,
            **sql_ast,
            **near_duplicate,
        }
        
        result["strict_passed"] = not (
            result["has_database_leakage"]
            or result["has_question_leakage"]
            or result["has_sql_leakage"]
            or result["has_query_ir_leakage"]
            or result["has_parent_child_violations"]
            or result["has_schema_family_leakage"]
            or result["has_sql_ast_leakage"]
            or result["has_near_duplicate_leakage"]
        )
        result["passed"] = not (
            result["has_database_leakage"]
            or result["has_parent_child_violations"]
            or result["has_schema_family_leakage"]
        )
        return result

    @staticmethod
    def _text_overlap(
        splits: dict[str, list[dict[str, Any]]],
        key: str,
        flag_name: str,
        count_name: str,
    ) -> dict[str, Any]:
        train_values = {normalize_text(row.get(key)) for row in splits.get("train", []) if row.get(key)}
        other_values = {
            normalize_text(row.get(key))
            for split_name in _comparison_split_names(splits)
            for row in splits.get(split_name, [])
            if row.get(key)
        }
        shared = {value for value in train_values & other_values if value}
        return {flag_name: bool(shared), count_name: len(shared), f"{key}_overlap": sorted(shared)}

    @staticmethod
    def _canonical_query_ir_signature(query_ir: Any) -> str | None:
        if not isinstance(query_ir, dict):
            return None
        scrubbed = json.loads(json.dumps(query_ir, sort_keys=True, default=str))
        if isinstance(scrubbed.get("metadata"), dict):
            scrubbed["metadata"] = {
                key: value
                for key, value in scrubbed["metadata"].items()
                if key not in {"split", "internal_split", "source_split"}
            }
        payload = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_generic_template_overlap(
        normalized_question: str,
        train_row: dict[str, Any],
        other_row: dict[str, Any],
    ) -> bool:
        if not _looks_like_generic_table_question(normalized_question):
            return False
        train_schema = _schema_fingerprint(train_row)
        other_schema = _schema_fingerprint(other_row)
        train_db = train_row.get("database_id") or train_row.get("db_id")
        other_db = other_row.get("database_id") or other_row.get("db_id")
        return bool(train_db and other_db and train_db != other_db and train_schema != other_schema)

    @staticmethod
    def _is_generic_sql_overlap(train_row: dict[str, Any], other_row: dict[str, Any]) -> bool:
        if not (
            _looks_like_generic_count_sql(str(train_row.get("source_sql") or ""))
            and _looks_like_generic_count_sql(str(other_row.get("source_sql") or ""))
        ):
            return False
        train_schema = _schema_fingerprint(train_row)
        other_schema = _schema_fingerprint(other_row)
        train_db = train_row.get("database_id") or train_row.get("db_id")
        other_db = other_row.get("database_id") or other_row.get("db_id")
        return bool(train_db and other_db and train_db != other_db and train_schema != other_schema)


def _comparison_split_names(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    preferred = [
        "validation",
        "development_validation",
        "model_selection_validation",
        "test",
        "frozen_semantic_test",
        "unseen_db_test",
        "unseen_database_test",
        "controlled_execution_test",
    ]
    return [name for name in preferred if name in splits]


def _schema_fingerprint(row: dict[str, Any]) -> str:
    schema = row.get("schema") or row.get("serialized_schema") or {}
    payload = json.dumps(schema, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _looks_like_generic_table_question(normalized_question: str) -> bool:
    tokens = normalized_question.split()
    if not tokens:
        return False
    if tokens[0] in {"list", "find", "get"} and len(tokens) <= 4:
        return True
    if tokens[0] == "show" and len(tokens) <= 5:
        return True
    if tokens[0] == "count" and len(tokens) <= 6:
        return True
    if tokens[:2] == ["how", "many"] and len(tokens) <= 6:
        return True
    if "number" in tokens and len(tokens) <= 6:
        return True
    return False


def _looks_like_generic_count_sql(sql: str) -> bool:
    text = normalize_text(sql).rstrip(";")
    return bool(
        re.fullmatch(
            r"select\s+count\s*\(\s*\*\s*\)\s+from\s+[`\"\[]?[a-z_][\w.]*[`\"\]]?",
            text,
            flags=re.IGNORECASE,
        )
    )
