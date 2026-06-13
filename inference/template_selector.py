from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .prediction_models import RetrievedCandidate


class TemplateSelector:
    def select_template(self, candidates: list[RetrievedCandidate], question: str) -> dict[str, Any]:
        if not candidates:
            return {
                "template_id": None,
                "intent": None,
                "confidence": 0.0,
                "reason": "no candidates retrieved",
                "candidate_votes": {},
            }
        q_template = self._question_template(question)
        top = candidates[0]
        votes = Counter(self._canonical_template(item.template_id, question) for item in candidates[:3])
        votes = Counter({key: value for key, value in votes.items() if key})
        if q_template:
            template_id = q_template
            reason = "question keyword rule"
            confidence = max(float(top.rerank_score or 0.0), 0.72)
        elif votes and votes.most_common(1)[0][1] >= 2:
            template_id = votes.most_common(1)[0][0]
            reason = "top candidate voting"
            confidence = min(1.0, float(top.rerank_score or 0.0) + 0.08)
        else:
            template_id = self._canonical_template(top.template_id, question)
            reason = "top reranked candidate"
            confidence = float(top.rerank_score or 0.0)

        if confidence < 0.55:
            reason += "; low confidence"
        return {
            "template_id": template_id,
            "intent": template_id,
            "confidence": round(confidence, 4),
            "reason": reason,
            "candidate_votes": dict(votes),
        }

    @staticmethod
    def _question_template(question: str) -> str | None:
        q = question.lower()
        has_filter_phrase = re.search(r"\b(where|with|excluding)\b|\bfor\s+(region|status|category)\b|\bin\s+(region|status|category)\b", q) is not None
        has_metric_word = any(word in q for word in ["sales", "revenue", "profit", "quantity", "average", "amount"])
        if any(word in q for word in ["top", "highest", "best", "most"]):
            return "top_n_metric_by_dimension"
        if any(word in q for word in ["bottom", "lowest", "least", "worst"]):
            return "bottom_n_metric_by_dimension"
        if any(word in q for word in ["how many", "count", "number of"]):
            return "count_by_dimension" if re.search(r"\bby\b", q) else "count_records"
        if any(phrase in q for phrase in ["last month", "this month", "last year", "this year", "last 30 days"]) and has_metric_word:
            return "metric_summary"
        if any(word in q for word in ["by month", "monthly", "by year", "yearly", "trend"]):
            return "trend_by_date"
        if has_filter_phrase and not has_metric_word:
            return "simple_filter"
        if re.search(r"\bby\s+\w+", q):
            return "metric_by_dimension"
        if has_filter_phrase and has_metric_word:
            return "metric_summary"
        return None

    @classmethod
    def _canonical_template(cls, template_id: str | None, question: str) -> str | None:
        if template_id == "rank_dimension":
            return cls._question_template(question) or "top_n_metric_by_dimension"
        if template_id == "aggregate_metric":
            return "metric_summary"
        if template_id == "count_dimension":
            return "count_by_dimension"
        if template_id == "time_series_metric":
            return "trend_by_date"
        if template_id == "detail_rows":
            return "show_records"
        if template_id == "filtered_rank_dimension":
            return "simple_filter" if "where" in question.lower() else cls._question_template(question) or "top_n_metric_by_dimension"
        if template_id == "distinct_dimension_values":
            return "show_records"
        if template_id == "metric_distribution":
            return "metric_summary"
        if template_id in {
            "show_records",
            "count_records",
            "count_by_dimension",
            "metric_summary",
            "metric_by_dimension",
            "top_n_metric_by_dimension",
            "bottom_n_metric_by_dimension",
            "trend_by_date",
            "simple_filter",
        }:
            return template_id
        return cls._question_template(question) or template_id
