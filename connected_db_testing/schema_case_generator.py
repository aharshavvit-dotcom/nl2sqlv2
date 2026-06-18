from __future__ import annotations

import json
import re
from typing import Any

from semantic_layer import build_semantic_profile
from semantic_layer.schema_profiler import schema_fingerprint, singularize, tokenize


class SchemaCaseGenerator:
    def generate_cases(self, schema: dict[str, Any], max_tables: int | None = None) -> list[dict[str, Any]]:
        profile = build_semantic_profile(schema)
        fingerprint = schema_fingerprint(schema)
        cases: list[dict[str, Any]] = []
        table_names = list((profile.get("tables") or {}).keys())
        if max_tables is not None:
            table_names = table_names[:max_tables]

        for table in table_names:
            info = profile["tables"][table]
            table_phrase = table.replace("_", " ")
            sensitive_terms = list(info.get("sensitive_columns") or [])
            cases.extend(
                [
                    self._direct_case(f"list_{_slug(table)}", f"list all {table_phrase}", table, fingerprint, sensitive_terms, "direct_table_listing"),
                    self._direct_case(f"show_{_slug(table)}", f"show {table_phrase}", table, fingerprint, sensitive_terms, "direct_table_listing"),
                    self._count_case(f"count_{_slug(table)}", f"count {table_phrase}", table, fingerprint, sensitive_terms),
                ]
            )
            for column in info.get("likely_filters", [])[:2]:
                cases.append(self._filter_case(f"filter_{_slug(table)}_{_slug(column)}", f"show {table_phrase} where {column.replace('_', ' ')} is active", table, column, fingerprint, sensitive_terms))

        cases.extend(self._relationship_cases(profile, fingerprint))
        return cases

    @staticmethod
    def _direct_case(case_id: str, question: str, table: str, fingerprint: str, sensitive_terms: list[str], case_type: str) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "question": question,
            "expected": {
                "source_model": "generic_direct_planner",
                "base_table": table,
                "must_include": [f'FROM "{table}"'],
                "must_not_include": ["JOIN", *sensitive_terms],
                "must_have_limit": True,
                "must_not_select_star": True,
            },
            "schema_fingerprint": fingerprint,
            "case_type": case_type,
        }

    @staticmethod
    def _count_case(case_id: str, question: str, table: str, fingerprint: str, sensitive_terms: list[str]) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "question": question,
            "expected": {
                "source_model": "generic_direct_planner",
                "base_table": table,
                "must_include": [f'FROM "{table}"', "COUNT"],
                "must_not_include": ["JOIN", *sensitive_terms],
                "must_have_limit": True,
                "must_not_select_star": False,
            },
            "schema_fingerprint": fingerprint,
            "case_type": "count_query",
        }

    @staticmethod
    def _filter_case(case_id: str, question: str, table: str, column: str, fingerprint: str, sensitive_terms: list[str]) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "question": question,
            "expected": {
                "source_model": "generic_direct_planner",
                "base_table": table,
                "filter_column": column,
                "must_include": [f'FROM "{table}"', f'"{column}"'],
                "must_not_include": ["JOIN", *sensitive_terms],
                "must_have_limit": True,
                "must_not_select_star": True,
            },
            "schema_fingerprint": fingerprint,
            "case_type": "filter_query",
        }

    @staticmethod
    def _relationship_cases(profile: dict[str, Any], fingerprint: str) -> list[dict[str, Any]]:
        cases = []
        for rel in profile.get("relationships", []):
            from_table = rel.get("from_table")
            to_table = rel.get("to_table")
            if not from_table or not to_table:
                continue
            from_phrase = from_table.replace("_", " ")
            target_tokens = [singularize(token) for token in tokenize(to_table)]
            target_phrase = " ".join(target_tokens) or to_table.replace("_", " ")
            cases.append(
                {
                    "case_id": f"join_{_slug(from_table)}_with_{_slug(to_table)}",
                    "question": f"show {from_phrase} with {target_phrase} names",
                    "expected": {
                        "base_table": from_table,
                        "must_include": [f'FROM "{from_table}"', "JOIN"],
                        "must_not_include": [],
                        "must_have_limit": True,
                        "must_not_select_star": False,
                    },
                    "schema_fingerprint": fingerprint,
                    "case_type": "explicit_join",
                    "relationship": rel,
                }
            )
        return cases


def write_cases_jsonl(path: str, cases: list[dict[str, Any]]) -> None:
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + ("\n" if cases else ""), encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())).strip("_")
