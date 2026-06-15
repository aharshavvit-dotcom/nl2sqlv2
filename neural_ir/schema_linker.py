from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .tokenizer import tokenize

try:  # pragma: no cover - optional dependency is present in the app requirements
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "sales": ["revenue", "amount"],
    "revenue": ["amount", "sales"],
    "customer": ["customers", "client", "guest", "customer_name"],
    "customers": ["customer", "client", "guest", "customer_name"],
    "product": ["products", "item", "sku", "product_name"],
    "products": ["product", "item", "sku", "product_name"],
    "month": ["date", "order_date", "transaction_date"],
    "year": ["date", "order_date", "transaction_date"],
    "completed": ["status"],
    "region": ["region"],
    "category": ["category"],
    "status": ["status"],
}


class SchemaLinker:
    def __init__(self, synonyms_path: str | Path | None = None):
        self.synonyms = _load_synonyms(synonyms_path)

    def link(self, question: str, candidates: dict[str, Any]) -> dict[str, Any]:
        question_tokens = _expand_tokens(tokenize(question), self.synonyms)
        question_terms = set(question_tokens)
        table_scores: dict[str, float] = {}
        column_scores: dict[str, float] = {}
        matched_terms: list[dict[str, Any]] = []

        for table in candidates.get("tables", []):
            score, terms = _score_candidate(question_terms, table.get("tokens", []), table.get("display", ""))
            table_scores[table["table"]] = score
            if terms:
                matched_terms.append({"candidate": table["display"], "terms": terms, "score": score})

        for column in candidates.get("columns", []):
            score, terms = _score_candidate(question_terms, column.get("tokens", []), column.get("display", ""))
            role_bonus = _role_bonus(question_terms, column)
            score = min(1.0, max(0.0, score + role_bonus - _id_penalty(question_terms, column)))
            key = column["display"]
            column_scores[key] = score
            if terms or role_bonus:
                matched_terms.append({"candidate": key, "terms": terms, "score": score})

        top_tables = [
            {**table, "score": table_scores.get(table["table"], 0.0)}
            for table in candidates.get("tables", [])
        ]
        top_columns = [
            {**column, "score": column_scores.get(column["display"], 0.0)}
            for column in candidates.get("columns", [])
        ]
        top_tables.sort(key=lambda item: (-item["score"], item["index"]))
        top_columns.sort(key=lambda item: (-item["score"], item["index"]))
        return {
            "table_scores": table_scores,
            "column_scores": column_scores,
            "top_tables": top_tables,
            "top_columns": top_columns,
            "debug": {"matched_terms": matched_terms, "question_tokens": question_tokens},
        }


def _load_synonyms(path: str | Path | None) -> dict[str, list[str]]:
    synonyms = {key: list(values) for key, values in DEFAULT_SYNONYMS.items()}
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(Path(__file__).resolve().parents[1] / "data" / "synonyms.yaml")
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        for section in ["metrics", "dimensions", "dates", "entities"]:
            for key, value in (payload.get(section) or {}).items():
                values = [str(key)]
                if isinstance(value, dict):
                    for field in ["aliases", "synonyms", "candidate_columns", "candidate_tables"]:
                        raw = value.get(field) or []
                        values.extend(str(item) for item in raw)
                    if value.get("column"):
                        values.append(str(value["column"]))
                    if value.get("expression"):
                        values.extend(tokenize(str(value["expression"]).replace("_", " ")))
                synonyms.setdefault(str(key), [])
                synonyms[str(key)].extend(values)
                for alias in values:
                    for token in tokenize(alias.replace("_", " ")):
                        synonyms.setdefault(token, [])
                        synonyms[token].extend(values)
        break
    return {key: sorted(set(tokenize(" ".join(values)))) for key, values in synonyms.items()}


def _expand_tokens(tokens: list[str], synonyms: dict[str, list[str]]) -> list[str]:
    expanded = []
    for token in tokens:
        expanded.append(token)
        expanded.append(_singular(token))
        expanded.extend(synonyms.get(token, []))
        expanded.extend(synonyms.get(_singular(token), []))
    return list(dict.fromkeys(expanded))


def _score_candidate(question_terms: set[str], candidate_tokens: list[str], display: str) -> tuple[float, list[str]]:
    tokens = set(_expand_tokens(candidate_tokens, DEFAULT_SYNONYMS))
    overlap = sorted(question_terms & tokens)
    overlap_score = len(overlap) / max(len(tokens), 1)
    substring_score = 0.0
    display_text = display.lower().replace("_", " ")
    for term in question_terms:
        if len(term) > 2 and term in display_text:
            substring_score = max(substring_score, 0.45)
    fuzzy_score = 0.0
    if fuzz is not None:
        for term in question_terms:
            fuzzy_score = max(fuzzy_score, float(fuzz.partial_ratio(term, display_text)) / 100.0 * 0.35)
    score = min(1.0, max(overlap_score, substring_score, fuzzy_score) + min(0.35, 0.12 * len(overlap)))
    return score, overlap


def _role_bonus(question_terms: set[str], column: dict[str, Any]) -> float:
    column_name = str(column.get("column") or "").lower()
    column_type = column.get("type")
    if column_type == "date" and question_terms & {"date", "month", "year", "monthly", "yearly"}:
        return 0.35
    if column_type == "numeric" and question_terms & {"sales", "revenue", "amount", "total", "price", "quantity", "fare", "count"}:
        return 0.20
    if column_name.endswith("_name") and question_terms & {"customer", "product", "store", "rep", "name"}:
        return 0.35
    if column_name in {"status", "region", "category"} and column_name in question_terms:
        return 0.25
    return 0.0


def _id_penalty(question_terms: set[str], column: dict[str, Any]) -> float:
    column_name = str(column.get("column") or "").lower()
    if column.get("type") == "id" and "id" not in question_terms and "identifier" not in question_terms:
        return 0.40
    if column_name.endswith("_id") and "id" not in question_terms:
        return 0.35
    return 0.0


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token
