from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Literal
from pydantic import BaseModel

from inference.grounding.filter_value_contract import (
    QueryTimeContext,
    ExtractedLiteral,
)
from inference.grounding.schema_value_index import SchemaValueIndex
from inference.grounding.value_normalizer import ValueNormalizer


class FilterValueExtractionContract(BaseModel):
    raw_question: str
    extracted_literals: list[ExtractedLiteral]


class FilterValueExtractor:
    def __init__(self, value_index: SchemaValueIndex):
        self.value_index = value_index

    def extract_literals(self, question: str, time_context: QueryTimeContext | None = None, neural_spans: list[str] | None = None) -> FilterValueExtractionContract:
        if not time_context:
            time_context = QueryTimeContext(current_datetime=datetime.now())

        extracted = []
        patterns = [
            (r"['\"]([^'\"]+)['\"]", "quoted_string", "string"),
            (r"\b(?:named|called)\s+([A-Za-z][A-Za-z0-9'-]*(?:\s+[A-Za-z0-9'-]+){0,4})", "named_phrase", "string"),
            (r"\bseason\s+(?:was\s+)?(?:in\s+)?(\d{4})\b", "year", "year"),
            (r"\b(\d{4}-\d{2}-\d{2})\b", "date", "date"),
            (r"\b(?:above|greater than|more than)\s*(?:₹|\$)?\s*(-?\d+(?:\.\d+)?)\s*(?:lakh|million|crore|%)?\b", "numeric_limit", "decimal"),
            (r"\b(?:below|less than)\s*(?:₹|\$)?\s*(-?\d+(?:\.\d+)?)\s*(?:lakh|million|crore|%)?\b", "numeric_limit", "decimal"),
            (r"\b(?:between)\s*(?:₹|\$)?\s*(-?\d+(?:\.\d+)?)\s*(?:lakh|million|crore)?\s*and\s*(?:₹|\$)?\s*(-?\d+(?:\.\d+)?)\s*(?:lakh|million|crore)?\b", "numeric_range", "range"),
            (r"\bin\s+([A-Za-z0-9' -]+(?:,\s*[A-Za-z0-9' -]+)*(?:\s*(?:or|and)\s+[A-Za-z0-9' -]+)?)\b", "list_values", "list"),
            (r"\bof\s+(?:the\s+)?([A-Z][\w'-]*(?:\s+[A-Z0-9][\w'-]*){0,4})[?.!]*$", "capitalized_entity", "string"),
        ]

        found_spans = []
        literal_counter = 0

        # Check for sample value exact matches from the index first to give them higher priority
        if self.value_index and getattr(self.value_index, "mode", None) != "disabled" and isinstance(getattr(self.value_index, "index", None), dict):
            indexed_keys = sorted(self.value_index.index.keys(), key=len, reverse=True)
            for key in indexed_keys:
                if not key or len(key) < 2:
                    continue
                words = key.split()
                pattern_parts = [re.escape(w) for w in words]
                pattern_str = r"\b" + r"\s*[^a-zA-Z0-9]*\s*".join(pattern_parts) + r"\b"
                try:
                    for match in re.finditer(pattern_str, question, re.IGNORECASE):
                        start, end = match.span(0)
                        if any(start < s_end and end > s_start for s_start, s_end in found_spans):
                            continue
                        found_spans.append((start, end))
                        
                        raw_text = match.group(0).strip()
                        extracted.append(ExtractedLiteral(
                            literal_id=f"lit_{literal_counter}",
                            raw_text=raw_text,
                            normalized_value=raw_text,
                            value_type="string",
                            span_start=start,
                            span_end=end,
                            extraction_method="index_exact_match",
                            extraction_confidence=0.95,
                        ))
                        literal_counter += 1
                except re.error:
                    continue

        for pattern, method, val_type in patterns:
            flags = re.IGNORECASE if method != "capitalized_entity" else 0
            for match in re.finditer(pattern, question, flags):
                start, end = match.span(0)
                if any(start < s_end and end > s_start for s_start, s_end in found_spans):
                    continue
                found_spans.append((start, end))

                raw_text = match.group(0).strip()
                val_text = match.group(1).strip() if len(match.groups()) > 0 else raw_text

                norm_val = ValueNormalizer.normalize(val_text, val_type, time_context)

                extracted.append(ExtractedLiteral(
                    literal_id=f"lit_{literal_counter}",
                    raw_text=raw_text,
                    normalized_value=norm_val,
                    value_type=val_type,  # type: ignore
                    span_start=start,
                    span_end=end,
                    extraction_method=method,
                    extraction_confidence=0.88,
                ))
                literal_counter += 1

        for match in re.finditer(r"(?<!\w)-?\b\d+(?:\.\d+)?\b", question):
            start, end = match.span(0)
            if any(start < s_end and end > s_start for s_start, s_end in found_spans):
                continue
            prefix = question[max(0, start - 12):start].lower()
            if re.search(r"\b(?:top|first|limit)\s*$", prefix):
                continue

            raw_text = match.group(0).strip()
            val_type = "integer" if raw_text.isdigit() else "decimal"
            if len(raw_text) == 4 and raw_text.isdigit():
                val_type = "year"

            norm_val = ValueNormalizer.normalize(raw_text, val_type, time_context)
            extracted.append(ExtractedLiteral(
                literal_id=f"lit_{literal_counter}",
                raw_text=raw_text,
                normalized_value=norm_val,
                value_type=val_type,  # type: ignore
                span_start=start,
                span_end=end,
                extraction_method="fallback_regex",
                extraction_confidence=0.75,
            ))
            literal_counter += 1

        for phrase in ["last month", "this month", "last year", "this year", "past 30 days", "yesterday", "today"]:
            if phrase in question.lower():
                start = question.lower().find(phrase)
                end = start + len(phrase)
                if any(start < s_end and end > s_start for s_start, s_end in found_spans):
                    continue
                found_spans.append((start, end))

                norm_val = ValueNormalizer.normalize(phrase, "date", time_context)
                extracted.append(ExtractedLiteral(
                    literal_id=f"lit_{literal_counter}",
                    raw_text=phrase,
                    normalized_value=norm_val,
                    value_type="date",
                    span_start=start,
                    span_end=end,
                    extraction_method="date_phrase",
                    extraction_confidence=0.85,
                ))
                literal_counter += 1

        # Integrate and arbitrate neural predicted spans
        if neural_spans:
            for span_text in neural_spans:
                span_text = span_text.strip()
                if not span_text:
                    continue
                try:
                    for match in re.finditer(re.escape(span_text), question, re.IGNORECASE):
                        start, end = match.span(0)
                        
                        # Agreement check: if overlaps with an already extracted literal, boost confidence
                        agreed = False
                        for lit in extracted:
                            if max(lit.span_start, start) < min(lit.span_end, end):
                                lit.extraction_confidence = min(0.99, lit.extraction_confidence + 0.10)
                                if "neural" not in lit.extraction_method:
                                    lit.extraction_method = f"{lit.extraction_method}+neural"
                                agreed = True
                        
                        if agreed:
                            continue
                            
                        # If deterministic missing, use neural span
                        found_spans.append((start, end))
                        raw_text = match.group(0).strip()
                        val_type = "string"
                        if raw_text.isdigit():
                            val_type = "integer" if len(raw_text) != 4 else "year"
                        elif re.match(r"^\d{4}-\d{2}-\d{2}$", raw_text):
                            val_type = "date"
                            
                        norm_val = ValueNormalizer.normalize(raw_text, val_type, time_context)
                        extracted.append(ExtractedLiteral(
                            literal_id=f"lit_{literal_counter}",
                            raw_text=raw_text,
                            normalized_value=norm_val,
                            value_type=val_type,  # type: ignore
                            span_start=start,
                            span_end=end,
                            extraction_method="neural_span",
                            extraction_confidence=0.90,
                        ))
                        literal_counter += 1
                except re.error:
                    continue

        return FilterValueExtractionContract(
            raw_question=question,
            extracted_literals=extracted,
        )
