from __future__ import annotations

from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
import re
from typing import Any

try:  # pragma: no cover - exercised when optional dependency is absent
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

from nl2sql_v1.schema import SchemaGraph

from .schema_text_normalizer import (
    column_name_variants,
    normalize_identifier,
    normalize_table_phrase,
    singularize_simple,
    table_name_variants,
    tokenize_identifier,
)


SENSITIVE_PATTERNS = (
    "password",
    "token",
    "secret",
    "ssn",
    "email",
    "phone",
    "address",
    "dob",
    "birth_date",
    "credit_card",
    "api_key",
    "auth",
)


class SchemaProfile:
    def __init__(self, schema: dict[str, Any] | SchemaGraph):
        self.schema = schema
        self.dialect = self._detect_dialect(schema)
        self.tables = self._normalize_schema(schema)
        self._table_variants = {table: table_name_variants(table) for table in self.tables}
        self._column_variants = {
            table: {column["name"]: column_name_variants(column["name"]) for column in info["columns"]}
            for table, info in self.tables.items()
        }
        self._apply_semantic_aliases()

    def get_tables(self) -> list[str]:
        return sorted(self.tables)

    def get_columns(self, table: str) -> list[dict[str, Any]]:
        return list(self.tables.get(table, {}).get("columns", []))

    def find_table_matches(self, question: str) -> list[dict[str, Any]]:
        question_text = str(question or "").lower()
        question_phrase = normalize_table_phrase(question_text)
        question_identifier = normalize_identifier(question_text)
        question_tokens = set(tokenize_identifier(question_text))
        question_singular = {singularize_simple(token) for token in question_tokens}
        matches: list[dict[str, Any]] = []

        for table, variants in self._table_variants.items():
            best: dict[str, Any] | None = None
            table_identifier = normalize_identifier(table)
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(table_identifier)}(?![A-Za-z0-9_])", question_identifier):
                best = self._match(table, 1.0, table_identifier, "exact_table_name")

            for variant in sorted(variants, key=len, reverse=True):
                if not variant:
                    continue
                variant_phrase = normalize_table_phrase(variant)
                if variant_phrase and re.search(rf"\b{re.escape(variant_phrase)}\b", question_phrase):
                    candidate = self._match(table, 1.0, variant_phrase, "exact_phrase")
                    best = self._better(best, candidate)
                    continue
                variant_identifier = normalize_identifier(variant)
                if variant_identifier and re.search(rf"(?<![A-Za-z0-9_]){re.escape(variant_identifier)}(?![A-Za-z0-9_])", question_identifier):
                    candidate = self._match(table, 0.98, variant_identifier, "identifier_phrase")
                    best = self._better(best, candidate)

            table_tokens = set(tokenize_identifier(table))
            table_singular = {singularize_simple(token) for token in table_tokens}
            overlap = (table_tokens | table_singular) & (question_tokens | question_singular)
            if overlap and table_tokens:
                score = min(0.92, 0.55 + (len(overlap) / len(table_tokens)) * 0.35)
                candidate = self._match(table, score, " ".join(sorted(overlap)), "token_overlap")
                best = self._better(best, candidate)

            fuzzy_score = self._fuzzy_best(question_phrase, variants)
            if fuzzy_score >= 0.88:
                candidate = self._match(table, fuzzy_score, table, "fuzzy")
                best = self._better(best, candidate)

            if best and best["score"] >= 0.55:
                matches.append(best)

        return sorted(matches, key=lambda item: (-item["score"], item["table"]))

    def find_column_matches(self, question: str, table: str | None = None) -> list[dict[str, Any]]:
        tables = [table] if table else self.get_tables()
        question_text = str(question or "").lower()
        question_phrase = normalize_table_phrase(question_text)
        question_identifier = normalize_identifier(question_text)
        question_tokens = set(tokenize_identifier(question_text))
        matches: list[dict[str, Any]] = []

        for table_name in tables:
            if table_name not in self.tables:
                continue
            for column in self.get_columns(table_name):
                column_name = column["name"]
                variants = self._column_variants.get(table_name, {}).get(column_name, column_name_variants(column_name))
                best: dict[str, Any] | None = None
                for variant in sorted(variants, key=len, reverse=True):
                    phrase = normalize_table_phrase(variant)
                    ident = normalize_identifier(variant)
                    if phrase and re.search(rf"\b{re.escape(phrase)}\b", question_phrase):
                        best = self._column_match(table_name, column_name, 1.0, phrase, "exact_phrase")
                        break
                    if ident and re.search(rf"(?<![A-Za-z0-9_]){re.escape(ident)}(?![A-Za-z0-9_])", question_identifier):
                        best = self._column_match(table_name, column_name, 0.98, ident, "identifier_phrase")
                        break
                if best is None:
                    column_tokens = set(tokenize_identifier(column_name))
                    overlap = column_tokens & question_tokens
                    if overlap and column_tokens:
                        score = min(0.88, 0.50 + (len(overlap) / len(column_tokens)) * 0.35)
                        best = self._column_match(table_name, column_name, score, " ".join(sorted(overlap)), "token_overlap")
                if best is None:
                    fuzzy_score = self._fuzzy_best(question_phrase, variants)
                    if fuzzy_score >= 0.88:
                        best = self._column_match(table_name, column_name, min(fuzzy_score, 0.87), column_name, "fuzzy")
                if best and best["score"] >= 0.55:
                    matches.append(best)

        return sorted(matches, key=lambda item: (-item["score"], item["table"], item["column"]))

    def safe_select_columns(self, table: str, max_columns: int = 12) -> list[str]:
        columns = [column for column in self.get_columns(table) if not self.is_sensitive_column(column["name"])]
        if not columns:
            id_columns = [column["name"] for column in self.get_columns(table) if self._is_id_column(column["name"])]
            return id_columns[:max_columns]

        def rank(column: dict[str, Any]) -> tuple[int, str]:
            name = column["name"].lower()
            if column.get("is_primary_key") or column.get("primary_key"):
                return (0, name)
            if name in {"name", "title", "label", "code", "status"}:
                return (1, name)
            if any(marker in name for marker in ("name", "code", "status", "type", "date", "created", "updated")):
                return (2, name)
            if self._is_id_column(name):
                return (4, name)
            return (3, name)

        return [column["name"] for column in sorted(columns, key=rank)[:max_columns]]

    def is_sensitive_column(self, column_name: str) -> bool:
        name = normalize_identifier(column_name)
        return any(pattern in name for pattern in SENSITIVE_PATTERNS)

    def _apply_semantic_aliases(self) -> None:
        try:
            from semantic_layer.schema_profiler import column_aliases_for, table_aliases_for
        except Exception:
            return
        for table in self.tables:
            self._table_variants.setdefault(table, set()).update(table_aliases_for(table))
            for column in self.get_columns(table):
                self._column_variants.setdefault(table, {}).setdefault(column["name"], set()).update(column_aliases_for(column["name"]))

    @staticmethod
    def _match(table: str, score: float, matched_text: str, match_type: str) -> dict[str, Any]:
        return {
            "table": table,
            "score": round(float(score), 4),
            "matched_text": matched_text,
            "match_type": match_type,
        }

    @staticmethod
    def _column_match(table: str, column: str, score: float, matched_text: str, match_type: str) -> dict[str, Any]:
        return {
            "table": table,
            "column": column,
            "score": round(float(score), 4),
            "matched_text": matched_text,
            "match_type": match_type,
        }

    @staticmethod
    def _better(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
        if current is None or candidate["score"] > current["score"]:
            return candidate
        return current

    @staticmethod
    def _fuzzy_best(question_phrase: str, variants: set[str]) -> float:
        if not question_phrase or not variants:
            return 0.0
        scores = []
        for variant in variants:
            phrase = normalize_table_phrase(variant)
            if not phrase:
                continue
            if fuzz is not None:
                scores.append(float(fuzz.partial_ratio(phrase, question_phrase)) / 100.0)
            else:
                scores.append(SequenceMatcher(None, phrase, question_phrase).ratio())
        return max(scores or [0.0])

    @staticmethod
    def _is_id_column(column_name: str) -> bool:
        name = column_name.lower()
        return name == "id" or name.endswith("_id") or name.endswith("_key")

    @staticmethod
    def _detect_dialect(schema: dict[str, Any] | SchemaGraph) -> str:
        if isinstance(schema, SchemaGraph):
            return getattr(schema, "dialect", "sqlite") or "sqlite"
        return str(schema.get("dialect") or "sqlite").lower()

    @staticmethod
    def _normalize_schema(schema: dict[str, Any] | SchemaGraph) -> dict[str, dict[str, Any]]:
        if isinstance(schema, SchemaGraph):
            normalized: dict[str, dict[str, Any]] = {}
            for table_name, table in schema.tables.items():
                columns = []
                for column_name, column in table.columns.items():
                    columns.append(
                        {
                            "name": column_name,
                            "type": str(column.type),
                            "is_primary_key": bool(column.primary_key),
                            "primary_key": bool(column.primary_key),
                        }
                    )
                normalized[table_name] = {"columns": columns}
            return normalized

        raw_tables = schema.get("tables", schema)
        normalized = {}
        for table_name, table_info in raw_tables.items():
            columns = table_info.get("columns", table_info) if isinstance(table_info, dict) else {}
            primary_keys = set(table_info.get("primary_keys", [])) if isinstance(table_info, dict) else set()
            if isinstance(columns, dict):
                iterable = columns.items()
            else:
                iterable = [(str(item.get("name", item)), item) for item in columns]
            normalized_columns = []
            for column_name, raw in iterable:
                raw_dict = asdict(raw) if is_dataclass(raw) else (raw if isinstance(raw, dict) else {})
                is_pk = bool(
                    raw_dict.get("is_primary_key")
                    or raw_dict.get("primary_key")
                    or column_name in primary_keys
                )
                normalized_columns.append(
                    {
                        **raw_dict,
                        "name": str(column_name),
                        "type": str(raw_dict.get("type", "")),
                        "is_primary_key": is_pk,
                        "primary_key": is_pk,
                    }
                )
            normalized[str(table_name)] = {"columns": normalized_columns}
        return normalized
