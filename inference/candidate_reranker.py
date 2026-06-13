from __future__ import annotations

import re

from rapidfuzz import fuzz

from .prediction_models import RetrievedCandidate
from .runtime_schema_context import RuntimeSchemaContext


class CandidateReranker:
    def rerank_candidates(
        self,
        question: str,
        candidates: list[RetrievedCandidate],
        schema_context: RuntimeSchemaContext,
    ) -> list[RetrievedCandidate]:
        for candidate in candidates:
            schema_score = self._schema_compatibility(question, candidate, schema_context)
            template_score = self._template_prior(question, candidate.template_id)
            slot_score = self._slot_detectability(question, candidate)
            rerank = (
                0.55 * candidate.similarity_score
                + 0.25 * schema_score
                + 0.10 * template_score
                + 0.10 * slot_score
            )
            candidate.schema_compatibility_score = round(schema_score, 4)
            candidate.rerank_score = round(rerank, 4)
        return sorted(candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)

    def _schema_compatibility(
        self,
        question: str,
        candidate: RetrievedCandidate,
        schema_context: RuntimeSchemaContext,
    ) -> float:
        q = question.lower()
        scores = []
        numeric_names = " ".join(schema_context.get_numeric_columns()).lower()
        text_names = " ".join(schema_context.get_text_columns()).lower()
        table_names = " ".join(schema_context.get_tables()).lower()
        date_names = " ".join(schema_context.get_date_columns()).lower()

        metric_slot = str(candidate.slots.get("metric") or "").lower()
        if metric_slot:
            metric_terms = [metric_slot, metric_slot.replace("_", " ")]
            scores.append(max(fuzz.partial_ratio(term, numeric_names) for term in metric_terms) / 100)
        if re.search(r"\bby\s+\w+", q):
            scores.append(fuzz.partial_ratio(q, text_names + " " + table_names) / 100)
        if any(word in q for word in ["month", "year", "date", "trend"]):
            scores.append(1.0 if date_names else 0.2)
        if not scores:
            scores.append(0.0)
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    @staticmethod
    def _template_prior(question: str, template_id: str | None) -> float:
        q = question.lower()
        desired = None
        if any(word in q for word in ["top", "highest", "best", "most"]):
            desired = "top_n_metric_by_dimension"
        elif any(word in q for word in ["bottom", "lowest", "least", "worst"]):
            desired = "bottom_n_metric_by_dimension"
        elif any(word in q for word in ["how many", "count", "number of"]):
            desired = "count_by_dimension" if re.search(r"\bby\b", q) else "count_records"
        elif any(word in q for word in ["by month", "monthly", "by year", "yearly", "trend"]):
            desired = "trend_by_date"
        elif re.search(r"\bby\s+\w+", q):
            desired = "metric_by_dimension"
        if not desired:
            return 0.3
        return 1.0 if template_id in {desired, "rank_dimension", "count_dimension"} else 0.35

    @staticmethod
    def _slot_detectability(question: str, candidate: RetrievedCandidate) -> float:
        q = question.lower()
        score = 0.0
        if any(word in q for word in ["sales", "revenue", "amount", "quantity", "orders", "count"]):
            score += 0.25
        if re.search(r"\bby\s+\w+", q) or any(word in q for word in ["customer", "product", "region", "status"]):
            score += 0.25
        if re.search(r"\b\d+\b", q) or candidate.slots.get("limit"):
            score += 0.15
        if candidate.slots:
            score += 0.10
        return min(score, 1.0)
