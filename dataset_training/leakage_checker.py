from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any

from .utils import normalize_text


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
            name: {str(row.get("db_id")) for row in rows if row.get("db_id")}
            for name, rows in splits.items()
            if name != "unsupported"
        }
        overlap: dict[str, list[str]] = {}
        for left, right in combinations(sorted(dbs), 2):
            shared = sorted(dbs[left] & dbs[right])
            if shared:
                overlap[f"{left}__{right}"] = shared
        train_unseen = sorted((dbs.get("train", set()) | dbs.get("validation", set()) | dbs.get("model_selection_validation", set())) & dbs.get("unseen_db_test", set()))
        return {
            "has_database_leakage": bool(overlap),
            "database_overlap": overlap,
            "train_unseen_overlap": train_unseen,
        }

    def check_question_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return self._text_overlap(splits, "question", "has_question_leakage", "question_overlap_count")

    def check_sql_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return self._text_overlap(splits, "source_sql", "has_sql_leakage", "sql_overlap_count")

    def check_parent_child_lineage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        id_to_split = {}
        for split_name, rows in splits.items():
            for row in rows:
                if "example_id" in row:
                    id_to_split[str(row["example_id"])] = split_name
                    
        violations = []
        for split_name, rows in splits.items():
            for row in rows:
                parent_id = (row.get("metadata") or {}).get("original_example_id")
                if parent_id and str(parent_id) in id_to_split:
                    parent_split = id_to_split[str(parent_id)]
                    if parent_split != split_name:
                        violations.append({
                            "example_id": row.get("example_id"),
                            "parent_id": parent_id,
                            "example_split": split_name,
                            "parent_split": parent_split
                        })
        return {
            "has_parent_child_violations": bool(violations),
            "parent_child_violations": violations
        }

    def check_schema_family_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        db_fingerprints = {}
        for split_name, rows in splits.items():
            for row in rows:
                db_id = row.get("db_id")
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
                db_id = row.get("db_id")
                if db_id and db_id in db_fingerprints:
                    split_to_fingerprints[split_name].add(db_fingerprints[db_id])
                    
        train_val_fps = split_to_fingerprints.get("train", set()) | split_to_fingerprints.get("validation", set()) | split_to_fingerprints.get("model_selection_validation", set())
        unseen_fps = split_to_fingerprints.get("unseen_db_test", set())
        
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
        train_semantic_sigs = set()
        for row in splits.get("train", []):
            sql = row.get("source_sql")
            if not sql:
                continue
            semantic_sig = get_semantic_ast_signature(sql, dialect)
            train_semantic_sigs.add(semantic_sig)
            
        violations = []
        for split_name in ["validation", "model_selection_validation", "test", "unseen_db_test"]:
            for row in splits.get(split_name, []):
                sql = row.get("source_sql")
                if not sql:
                    continue
                semantic_sig = get_semantic_ast_signature(sql, dialect)
                if semantic_sig in train_semantic_sigs:
                    violations.append({
                        "example_id": row.get("example_id"),
                        "question": row.get("question"),
                        "sql": sql,
                        "split": split_name,
                        "leakage_type": "exact_sql_ast_leakage"
                    })
        return {
            "has_sql_ast_leakage": bool(violations),
            "sql_ast_violations": violations
        }

    def check_near_duplicate_leakage(self, splits: dict[str, list[dict[str, Any]]], token_threshold: float = 0.85, char_threshold: float = 0.85) -> dict[str, Any]:
        train_by_intent = defaultdict(list)
        for row in splits.get("train", []):
            intent = row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"
            q = row.get("question")
            if q:
                train_by_intent[intent].append((row.get("example_id"), q, set(q.lower().split())))
                
        violations = []
        for split_name in ["validation", "model_selection_validation", "test", "unseen_db_test"]:
            for row in splits.get(split_name, []):
                intent = row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"
                q = row.get("question")
                if not q:
                    continue
                tokens = set(q.lower().split())
                for train_id, train_q, train_tokens in train_by_intent[intent]:
                    tok_sim = len(tokens & train_tokens) / max(len(tokens | train_tokens), 1)
                    if tok_sim >= token_threshold:
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
        parent_child = self.check_parent_child_lineage(splits)
        schema_family = self.check_schema_family_leakage(splits)
        sql_ast = self.check_sql_ast_leakage(splits)
        near_duplicate = self.check_near_duplicate_leakage(splits)
        
        result = {
            **database,
            **question,
            **sql,
            **parent_child,
            **schema_family,
            **sql_ast,
            **near_duplicate,
        }
        
        result["strict_passed"] = not (
            result["has_database_leakage"]
            or result["has_question_leakage"]
            or result["has_sql_leakage"]
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
            for split_name in ["validation", "model_selection_validation", "test", "unseen_db_test"]
            for row in splits.get(split_name, [])
            if row.get(key)
        }
        shared = {value for value in train_values & other_values if value}
        return {flag_name: bool(shared), count_name: len(shared), f"{key}_overlap": sorted(shared)}
