from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .prediction_models import RetrievedCandidate, RuntimeSlot
from .runtime_schema_context import RuntimeSchemaContext
from .synonym_loader import load_synonym_config, normalize_section


import hashlib

class SlotResolver:
    def __init__(
        self,
        filter_value_extractor: Any | None = None,
        grounding_service: Any | None = None,
    ):
        self._extractor = filter_value_extractor
        self._grounding_service = grounding_service
        self._cached_fingerprint = None

    def _schema_fingerprint(self, schema_context: RuntimeSchemaContext) -> str:
        tbls = sorted(schema_context.get_tables())
        hasher = hashlib.sha256()
        for t in tbls:
            hasher.update(t.encode("utf-8"))
            cols = sorted(schema_context.get_table_columns(t))
            for c in cols:
                hasher.update(c.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def resolve_slots(
        self,
        question: str,
        selected_template: dict[str, Any],
        candidates: list[RetrievedCandidate],
        schema_context: RuntimeSchemaContext,
        synonym_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        q = question.lower()
        synonyms = self._synonym_config(synonym_config, schema_context)
        slots: dict[str, RuntimeSlot] = {}
        slots["metric"] = self._metric_slot(q, candidates, selected_template, synonyms["metrics"])
        slots["dimension"] = self._dimension_slot(q, candidates, selected_template, synonyms["dimensions"])
        slots["entity"] = self._entity_slot(q, schema_context)
        slots["limit"] = self._limit_slot(q)
        slots["sort_direction"] = RuntimeSlot(
            slot_name="sort_direction",
            value="ASC" if any(word in q for word in ["bottom", "lowest", "least", "worst"]) else "DESC",
            source="question",
            confidence=0.9,
        )
        slots["date_grain"] = self._date_grain_slot(q)
        slots["date_filter"] = self._date_filter_slot(q)

        # Initialize extractor and grounding service lazily with caching
        if not self._extractor or not self._grounding_service:
            fingerprint = self._schema_fingerprint(schema_context)
            if self._cached_fingerprint != fingerprint:
                from inference.grounding.schema_value_index import SchemaValueIndex, ValueIndexMode
                from inference.grounding.filter_value_extractor import FilterValueExtractor
                from inference.grounding.filter_grounding_service import FilterGroundingService
                
                value_index = SchemaValueIndex(schema_context, mode=ValueIndexMode.APPROVED_DOMAIN_VALUES)
                self._extractor = FilterValueExtractor(value_index)
                self._grounding_service = FilterGroundingService(value_index, schema_context)
                self._cached_fingerprint = fingerprint

        grounded = False
        clar_questions = []
        filter_value_candidates = []

        if self._grounding_service and self._extractor:
            contract = self._extractor.extract_literals(question)
            gf_list = self._grounding_service.ground_filters(
                question, contract, entity_table=slots["entity"].value, metric_table=slots["metric"].value
            )
            for gf in gf_list:
                if gf.requires_clarification and gf.clarification_question:
                    clar_questions.append(gf.clarification_question)

            valid_gfs = [gf for gf in gf_list if gf.selected_candidate]
            if valid_gfs:
                valid_gfs.sort(key=lambda x: (-x.selected_candidate.grounding_score, -sum(x.selected_candidate.grounding_signals.values())))
                best_gf = valid_gfs[0]
                cand = best_gf.selected_candidate
                
                op_map = {
                    "equals": "equals",
                    "not_equals": "not_equals",
                    ">": "greater_than",
                    "<": "less_than",
                    ">=": "greater_than_or_equals",
                    "<=": "less_than_or_equals",
                    "between": "between",
                    "in": "in",
                    "not_in": "not_in",
                }
                slots["filter_column"] = RuntimeSlot(
                    slot_name="filter_column",
                    value=cand.column_name,
                    source="schema_match",
                    confidence=cand.grounding_score,
                    alternatives=[c.column_name for c in best_gf.candidate_columns[1:]],
                )
                slots["filter_value"] = RuntimeSlot(
                    slot_name="filter_value",
                    value=cand.normalized_value,
                    source="question",
                    confidence=0.9,
                )
                slots["filter_operator"] = RuntimeSlot(
                    slot_name="filter_operator",
                    value=op_map.get(cand.operator, cand.operator),
                    source="question",
                    confidence=0.88,
                )
                for gf in gf_list:
                    lit = next((l for l in contract.extracted_literals if l.literal_id == gf.literal_id), None)
                    if lit:
                        cand_cols = []
                        for c in gf.candidate_columns:
                            diag_signals = []
                            for sig_name in c.grounding_signals.keys():
                                if sig_name in ("exact_value_match", "fuzzy_value_match"):
                                    diag_signals.append("value_lookup")
                                else:
                                    diag_signals.append(sig_name)
                            if not diag_signals:
                                diag_signals = ["fallback"]
                            cand_cols.append({
                                "column": f"{c.table_name}.{c.column_name}",
                                "score": c.grounding_score,
                                "signals": diag_signals,
                            })
                        filter_value_candidates.append({
                            "value": lit.raw_text,
                            "span": [lit.span_start, lit.span_end],
                            "signals": [lit.extraction_method],
                            "candidate_columns": cand_cols,
                        })
                grounded = True

        if not grounded:
            filter_value_candidates = self.extract_filter_value_candidates(question, schema_context)
            filter_column, filter_value, filter_operator = self._filter_slots(
                question,
                q,
                synonyms["dimensions"],
                schema_context,
                filter_value_candidates,
            )
            slots["filter_column"] = filter_column
            slots["filter_value"] = filter_value
            slots["filter_operator"] = filter_operator

        template_id = selected_template.get("template_id")
        if template_id in {"count_records"} and not slots["metric"].value:
            slots["metric"] = RuntimeSlot(slot_name="metric", value="record_count", source="default", confidence=0.72)
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"} and not slots["dimension"].value:
            voted = self._candidate_vote(candidates, "dimension")
            if voted:
                slots["dimension"] = RuntimeSlot(slot_name="dimension", value=voted, source="retrieved_example", confidence=0.5)

        return {
            "slots": {key: value.model_dump() for key, value in slots.items()},
            "clarification_questions": clar_questions,
            "filter_value_candidates": filter_value_candidates,
        }

    def extract_filter_value_candidates(
        self,
        question: str,
        schema_context: RuntimeSchemaContext,
    ) -> list[dict[str, Any]]:
        """Extract literal/entity candidates and rank the columns they may filter."""
        patterns: list[tuple[str, str]] = [
            (r"['\"]([^'\"]+)['\"]", "quoted_string"),
            (r"\b(?:named|called)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,4})", "near_named_phrase"),
            (r"\bseason\s+(?:was\s+)?(?:in\s+)?(\d{4})\b", "year"),
            (r"\b(\d{4}-\d{2}-\d{2})\b", "date"),
            (r"\bof\s+(?:the\s+)?([A-Z][\w'-]*(?:\s+[A-Z0-9][\w'-]*){0,4})[?.!]*$", "capitalized_entity"),
        ]
        found: dict[tuple[int, int, str], set[str]] = {}
        for pattern, signal in patterns:
            for match in re.finditer(pattern, question):
                value = match.group(1).strip().rstrip("?.!,")
                span = match.span(1)
                if value:
                    found.setdefault((span[0], span[1], value), set()).add(signal)

        # Numeric literals and years not already captured. Limits introduced by
        # "top/first/limit" are intentionally excluded.
        for match in re.finditer(r"\b\d+(?:\.\d+)?\b", question):
            prefix = question[max(0, match.start() - 12):match.start()].lower()
            if re.search(r"\b(?:top|first|limit)\s*$", prefix):
                continue
            value = match.group(0)
            signal = "year" if len(value) == 4 and value.isdigit() else "numeric_value"
            found.setdefault((match.start(), match.end(), value), set()).add(signal)

        candidates = []
        for start, end, value in sorted(found, key=lambda item: (item[0], -(item[1] - item[0]))):
            ranked_columns = self._score_filter_columns(question, value, (start, end), schema_context)
            candidates.append({
                "value": value,
                "span": [start, end],
                "signals": sorted(found[(start, end, value)]),
                "candidate_columns": ranked_columns[:3],
            })
        return sorted(
            candidates,
            key=lambda item: -float((item.get("candidate_columns") or [{}])[0].get("score", 0.0)),
        )

    def _metric_slot(
        self,
        q: str,
        candidates: list[RetrievedCandidate],
        selected_template: dict[str, Any],
        metric_synonyms: dict[str, list[str]],
    ) -> RuntimeSlot:
        metric_aliases = [
            (metric, alias)
            for metric, aliases in metric_synonyms.items()
            for alias in self._aliases(metric, aliases)
        ]
        for metric, alias in sorted(metric_aliases, key=lambda item: len(item[1]), reverse=True):
            if self._contains_alias(q, alias):
                return RuntimeSlot(slot_name="metric", value=metric, source="question", confidence=0.92)
        voted = self._candidate_vote(candidates, "metric")
        if voted and voted not in {"*", "None"}:
            return RuntimeSlot(slot_name="metric", value=str(voted), source="retrieved_example", confidence=0.45)
        if selected_template.get("template_id") in {"count_records", "count_by_dimension"}:
            return RuntimeSlot(slot_name="metric", value="record_count", source="default", confidence=0.8)
        return RuntimeSlot(slot_name="metric", value=None, source="default", confidence=0.0)

    def _dimension_slot(
        self,
        q: str,
        candidates: list[RetrievedCandidate],
        selected_template: dict[str, Any],
        dimension_synonyms: dict[str, list[str]],
    ) -> RuntimeSlot:
        if "by month" in q or "monthly" in q:
            return RuntimeSlot(slot_name="dimension", value="month", source="question", confidence=0.95)
        if "by year" in q or "yearly" in q:
            return RuntimeSlot(slot_name="dimension", value="year", source="question", confidence=0.95)
        for dimension, aliases in dimension_synonyms.items():
            if any(self._contains_alias(q, alias) for alias in self._aliases(dimension, aliases)):
                return RuntimeSlot(slot_name="dimension", value=dimension, source="question", confidence=0.9)
        by_match = re.search(r"\bby\s+([a-z_ ]+)", q)
        if by_match:
            return RuntimeSlot(slot_name="dimension", value=by_match.group(1).strip(), source="question", confidence=0.65)
        voted = self._candidate_vote(candidates, "dimension")
        return RuntimeSlot(slot_name="dimension", value=voted, source="retrieved_example" if voted else "default", confidence=0.45 if voted else 0.0)

    @staticmethod
    def _entity_slot(q: str, schema_context: RuntimeSchemaContext) -> RuntimeSlot:
        for table in schema_context.get_tables():
            if table.lower() in q or table.lower().rstrip("s") in q:
                return RuntimeSlot(slot_name="entity", value=table, source="question", confidence=0.85)
        tables = schema_context.get_tables()
        return RuntimeSlot(slot_name="entity", value=tables[0] if tables else None, source="default", confidence=0.35)

    @staticmethod
    def _limit_slot(q: str) -> RuntimeSlot:
        match = re.search(r"\b(?:top|first|show|limit)?\s*(\d{1,4})\b", q)
        value = min(int(match.group(1)), 1000) if match else 100
        return RuntimeSlot(slot_name="limit", value=value, source="question" if match else "default", confidence=0.9 if match else 0.65)

    @staticmethod
    def _date_grain_slot(q: str) -> RuntimeSlot:
        if "month" in q or "monthly" in q:
            return RuntimeSlot(slot_name="date_grain", value="month", source="question", confidence=0.9)
        if "year" in q or "yearly" in q:
            return RuntimeSlot(slot_name="date_grain", value="year", source="question", confidence=0.9)
        return RuntimeSlot(slot_name="date_grain", value=None, source="default", confidence=0.0)

    @staticmethod
    def _date_filter_slot(q: str) -> RuntimeSlot:
        for phrase in ["last month", "this month", "last year", "this year", "last 30 days"]:
            if phrase in q:
                return RuntimeSlot(slot_name="date_filter", value=phrase, source="question", confidence=0.75)
        return RuntimeSlot(slot_name="date_filter", value=None, source="default", confidence=0.0)

    def _filter_slots(
        self,
        question: str,
        q: str,
        dimension_synonyms: dict[str, list[str]],
        schema_context: RuntimeSchemaContext,
        filter_value_candidates: list[dict[str, Any]] | None = None,
    ) -> tuple[RuntimeSlot, RuntimeSlot, RuntimeSlot]:
        stop_words = {
            "limit",
            "order",
            "group",
            "by",
            "from",
            "of",
            "with",
            "where",
            "show",
            "list",
            "display",
            "records",
            "rows",
            "orders",
            "customers",
            "products",
            "stores",
        }
        candidates: list[tuple[int, str, str, str]] = []
        if "excluding " in q:
            excluded = q.split("excluding ", 1)[1].strip().split()[0]
            return (
                RuntimeSlot(slot_name="filter_column", value=None, source="default", confidence=0.0),
                RuntimeSlot(slot_name="filter_value", value=excluded, source="question", confidence=0.4),
                RuntimeSlot(slot_name="filter_operator", value="not_equals", source="question", confidence=0.4),
            )
        grounded = (filter_value_candidates or [None])[0]
        if isinstance(grounded, dict) and grounded.get("candidate_columns"):
            ranked = grounded["candidate_columns"]
            best = ranked[0]
            alternatives = [
                item["column"] for item in ranked[1:]
                if float(item.get("score", 0.0)) >= float(best.get("score", 0.0)) - 0.08
                and float(item.get("score", 0.0)) >= 0.55
            ]
            confidence = float(best.get("score", 0.0))
            if alternatives:
                confidence = min(confidence, 0.49)
            if confidence >= 0.40:
                return (
                    RuntimeSlot(
                        slot_name="filter_column",
                        value=str(best["column"]).split(".", 1)[-1],
                        source="schema_match",
                        confidence=confidence,
                        alternatives=alternatives,
                    ),
                    RuntimeSlot(
                        slot_name="filter_value",
                        value=grounded["value"],
                        source="question",
                        confidence=0.9,
                    ),
                    RuntimeSlot(slot_name="filter_operator", value="equals", source="question", confidence=0.88),
                )
        standalone_value = self._standalone_filter_value(question)
        if standalone_value:
            inferred = self._infer_filter_column(q, standalone_value, schema_context)
            if inferred.get("column"):
                return (
                    RuntimeSlot(
                        slot_name="filter_column",
                        value=inferred["column"],
                        source="schema_match",
                        confidence=float(inferred["confidence"]),
                        alternatives=list(inferred.get("alternatives") or []),
                    ),
                    RuntimeSlot(slot_name="filter_value", value=standalone_value, source="question", confidence=0.88),
                    RuntimeSlot(slot_name="filter_operator", value="equals", source="question", confidence=0.88),
                )
        for dimension, aliases in dimension_synonyms.items():
            if dimension in {"month", "year"}:
                continue
            for alias in sorted(self._aliases(dimension, aliases), key=len, reverse=True):
                pattern = rf"(?:where|with|for|in|from)?\s*\b{re.escape(alias.lower())}\b\s*(is\s+not|was\s+not|!=|<>|equals|equal\s+to|was|is|=|as|in)?\s+([a-z0-9][a-z0-9 -]{{0,60}}?)(?=\s+(?:and|with|where|who|that|were|was\s+acquired)\b|[?.!,]|$)"
                match = re.search(pattern, q)
                if not match:
                    continue
                raw_operator = (match.group(1) or "").strip()
                raw_value = match.group(2).strip()
                words = []
                for word in raw_value.split():
                    if word in stop_words:
                        break
                    words.append(word)
                value = " ".join(words).strip()
                if value:
                    operator = "not_equals" if raw_operator in {"is not", "!=", "<>"} else "equals"
                    candidates.append((match.start(), dimension, value, operator))
        if candidates:
            _, dimension, value, operator = max(candidates, key=lambda item: item[0])
            return (
                RuntimeSlot(slot_name="filter_column", value=dimension, source="question", confidence=0.82),
                RuntimeSlot(slot_name="filter_value", value=value, source="question", confidence=0.82),
                RuntimeSlot(slot_name="filter_operator", value=operator, source="question", confidence=0.82),
            )

        return (
            RuntimeSlot(slot_name="filter_column", value=None, source="default", confidence=0.0),
            RuntimeSlot(slot_name="filter_value", value=None, source="default", confidence=0.0),
            RuntimeSlot(slot_name="filter_operator", value="equals", source="default", confidence=0.0),
        )

    @staticmethod
    def _standalone_filter_value(question: str) -> str | None:
        patterns = [
            r"\b(?:named|called)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,4})",
            r"\bseason\s+(?:was\s+)?(?:in\s+)?(\d{4})\b",
            r"\bof\s+(?:the\s+)?([A-Z][\w'-]*(?:\s+[A-Z0-9][\w'-]*){0,4})[?.!]*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1).strip().rstrip("?.!,")
        return None

    @staticmethod
    def _infer_filter_column(
        question: str,
        value: str,
        schema_context: RuntimeSchemaContext,
    ) -> dict[str, Any]:
        candidates: list[tuple[float, str, str]] = []
        value_lower = value.lower()
        for qualified in schema_context.get_columns():
            table, column = qualified.split(".", 1)
            info = schema_context.column_info(table, column)
            if info.get("is_sensitive"):
                continue
            score = 0.0
            method = "fallback"
            samples = [str(item).lower() for item in info.get("sample_values") or []]
            if value_lower in samples:
                score = 1.0
                method = "value_lookup"
            elif column.lower() in question or column.lower().replace("_", " ") in question:
                score = 0.9
                method = "exact"
            elif any(marker in column.lower() for marker in ("name", "player", "person", "model", "title")):
                score = 0.72
                method = "fuzzy"
            elif info.get("is_text"):
                score = 0.45
            if "player" in question and any(marker in column.lower() for marker in ("player", "name")):
                score += 0.15
            if table.lower().rstrip("s") in question:
                score += 0.08
            candidates.append((min(score, 1.0), qualified, method))
        if not candidates:
            return {"column": None, "confidence": 0.0, "method": "fallback", "alternatives": []}
        ordered = sorted(candidates, reverse=True)
        score, qualified, method = ordered[0]
        alternatives = [item[1] for item in ordered[1:4] if item[0] >= score - 0.08]
        if alternatives:
            score = min(score, 0.49)
        return {
            "column": qualified.split(".", 1)[1],
            "confidence": round(score, 4),
            "method": "ambiguous" if alternatives else method,
            "alternatives": alternatives,
        }

    @staticmethod
    def _score_filter_columns(
        question: str,
        value: str,
        span: tuple[int, int],
        schema_context: RuntimeSchemaContext,
    ) -> list[dict[str, Any]]:
        question_lower = question.lower()
        value_lower = value.lower().strip()
        value_normalized = re.sub(r"[^a-z0-9]+", " ", value_lower).strip()
        numeric_value = bool(re.fullmatch(r"\d+(?:\.\d+)?", value_lower))
        ranked: list[dict[str, Any]] = []
        for qualified in schema_context.get_columns():
            table, column = qualified.split(".", 1)
            info = schema_context.column_info(table, column)
            if info.get("is_sensitive"):
                continue
            signals: list[str] = []
            score = 0.0
            samples = [str(item).strip().lower() for item in info.get("sample_values") or []]
            normalized_samples = [re.sub(r"[^a-z0-9]+", " ", item).strip() for item in samples]
            if value_lower in samples:
                score, signals = 0.94, ["value_lookup", "exact_cell_value"]
            elif value_normalized and value_normalized in normalized_samples:
                score, signals = 0.88, ["value_lookup", "normalized_value_match"]

            column_phrase = column.lower().replace("_", " ")
            if column_phrase in question_lower:
                score += 0.18
                signals.append("column_context")
            column_position = question_lower.find(column_phrase)
            if column_position >= 0:
                distance = min(abs(span[0] - column_position), 80)
                score += max(0.0, 0.12 * (1.0 - distance / 80.0))
                signals.append("question_phrase_proximity")
            if numeric_value and info.get("is_numeric"):
                score += 0.12
                signals.append("type_compatible")
            elif not numeric_value and info.get("is_text"):
                score += 0.08
                signals.append("type_compatible")
            entity_markers = ("name", "player", "person", "model", "title", "code")
            if not numeric_value and any(marker in column.lower() for marker in entity_markers):
                score += 0.16
                signals.append("entity_column")
            context_tokens = set(re.findall(r"[a-z0-9]+", question_lower[max(0, span[0] - 30):span[1] + 30]))
            column_tokens = set(re.findall(r"[a-z0-9]+", column_phrase))
            if context_tokens & column_tokens:
                score += 0.10
                signals.append("token_overlap")
            fuzzy = SequenceMatcher(None, value_normalized, column_phrase).ratio()
            if fuzzy >= 0.70:
                score += 0.05 * fuzzy
                signals.append("fuzzy_match")
            ranked.append({
                "column": qualified,
                "score": round(min(score, 1.0), 4),
                "signals": list(dict.fromkeys(signals)) or ["fallback"],
            })
        return sorted(ranked, key=lambda item: (-float(item["score"]), item["column"]))

    @staticmethod
    def _candidate_vote(candidates: list[RetrievedCandidate], slot_name: str) -> str | None:
        values = [str(item.slots.get(slot_name)) for item in candidates if item.slots.get(slot_name)]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def _contains_alias(text: str, alias: str) -> bool:
        return re.search(rf"\b{re.escape(alias.lower())}\b", text) is not None

    @staticmethod
    def _aliases(key: str, aliases: list[str]) -> list[str]:
        return [key.replace("_", " "), key, *aliases]

    @staticmethod
    def _synonym_config(
        synonym_config: dict[str, Any] | None,
        schema_context: RuntimeSchemaContext,
    ) -> dict[str, dict[str, list[str]]]:
        from generic_planner.generic_slot_resolver import is_sample_retail_schema

        if synonym_config and (synonym_config.get("metrics") or synonym_config.get("dimensions")):
            configured = {
                "metrics": normalize_section(synonym_config.get("metrics") or {}),
                "dimensions": normalize_section(synonym_config.get("dimensions") or {}),
            }
        elif is_sample_retail_schema(schema_context.get_tables()):
            raw = load_synonym_config()
            configured = {
                "metrics": normalize_section(raw.get("metrics") or {}),
                "dimensions": normalize_section(raw.get("dimensions") or {}),
            }
        else:
            configured = {"metrics": {}, "dimensions": {}}

        # Connected schemas always contribute their own neutral vocabulary.  This
        # is the generic fallback; bundled retail terms are never needed for it.
        for qualified in schema_context.get_columns():
            table, column = qualified.split(".", 1)
            info = schema_context.column_info(table, column)
            if info.get("is_sensitive"):
                continue
            aliases = [column, column.replace("_", " "), f"{table} {column.replace('_', ' ')}"]
            configured["dimensions"].setdefault(column, aliases)
            if info.get("is_numeric") and not info.get("is_id"):
                configured["metrics"].setdefault(column, aliases)
        return configured
