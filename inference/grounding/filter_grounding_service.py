from __future__ import annotations

import re
from typing import Any
from rapidfuzz import fuzz

from inference.grounding.filter_value_contract import (
    ExtractedLiteral,
    GroundedFilterCandidate,
    GroundedFilter,
)
from inference.grounding.schema_value_index import SchemaValueIndex
from inference.runtime_schema_context import RuntimeSchemaContext


class FilterGroundingService:
    def __init__(self, value_index: SchemaValueIndex, schema_context: RuntimeSchemaContext):
        self.value_index = value_index
        self.schema_context = schema_context

    def ground_filters(
        self,
        question: str,
        contract: Any,
        entity_table: str | None = None,
        metric_table: str | None = None,
    ) -> list[GroundedFilter]:
        grounded_filters = []
        for literal in contract.extracted_literals:
            operator = self._extract_operator(question, literal)
            lookup_candidates = self.value_index.lookup_value(str(literal.normalized_value))

            if not lookup_candidates:
                lookup_candidates = self._generate_fallback_candidates(literal)

            ranked_candidates = []
            for item in lookup_candidates:
                column_qualified = item["column"]
                table, column = column_qualified.split(".", 1)

                base_score = item["score"]
                signals = dict(item.get("signals") or {})

                col_phrase = column.lower().replace("_", " ")
                if col_phrase in question.lower():
                    base_score += 0.18
                    signals["column_context"] = 0.18

                if table == entity_table:
                    base_score += 0.15
                    signals["active_entity_relevance"] = 0.15
                elif table == metric_table:
                    base_score += 0.10
                    signals["active_metric_relevance"] = 0.10

                if entity_table and self._is_fk_neighbor(table, entity_table):
                    base_score += 0.12
                    signals["join_graph_proximity"] = 0.12
                elif entity_table and table != entity_table:
                    if not self._is_reachable(table, entity_table):
                        base_score -= 0.50
                        signals["unreachable_table_penalty"] = -0.50

                final_score = round(max(0.0, min(1.0, base_score)), 4)

                ranked_candidates.append(GroundedFilterCandidate(
                    literal_id=literal.literal_id,
                    table_name=table,
                    column_name=column,
                    operator=operator,
                    normalized_value=literal.normalized_value,
                    grounding_score=final_score,
                    grounding_signals=signals,
                    ambiguity_score=0.0,
                ))

            ranked_candidates.sort(key=lambda x: (-x.grounding_score, -sum(x.grounding_signals.values())))

            selected = None
            requires_clar = False
            clar_question = None

            if ranked_candidates:
                top_cand = ranked_candidates[0]
                if len(ranked_candidates) > 1:
                    second_cand = ranked_candidates[1]
                    margin = top_cand.grounding_score - second_cand.grounding_score

                    for c in ranked_candidates:
                        c.ambiguity_score = round(1.0 - margin, 4)

                    if top_cand.grounding_score < 0.45:
                        requires_clar = True
                        clar_question = f"What field does '{literal.raw_text}' refer to?"
                    elif margin < 0.08:
                        requires_clar = True
                        clar_question = f"Does '{literal.raw_text}' refer to {top_cand.table_name}.{top_cand.column_name} or {second_cand.table_name}.{second_cand.column_name}?"
                else:
                    if top_cand.grounding_score < 0.45:
                        requires_clar = True
                        clar_question = f"What field does '{literal.raw_text}' refer to?"

                if not requires_clar:
                    selected = top_cand

            grounded_filters.append(GroundedFilter(
                literal_id=literal.literal_id,
                selected_candidate=selected,
                candidate_columns=ranked_candidates,
                requires_clarification=requires_clar,
                clarification_question=clar_question,
            ))

        return grounded_filters

    def _extract_operator(self, question: str, literal: ExtractedLiteral) -> str:
        start = max(0, literal.span_start - 25)
        end = min(len(question), literal.span_end + 25)
        context = question[start:end].lower()

        if "between" in context:
            return "between"
        if "not in" in context:
            return "not_in"
        if "in" in context and literal.value_type == "list":
            return "in"
        if "not equal" in context or "excluding" in context or "except" in context:
            return "not_equals"
        if "above" in context or "greater than" in context or "more than" in context:
            return ">"
        if "below" in context or "less than" in context:
            return "<"
        return "equals"

    def _generate_fallback_candidates(self, literal: ExtractedLiteral) -> list[dict[str, Any]]:
        candidates = []
        is_num = literal.value_type in ("integer", "decimal", "year")
        is_dt = literal.value_type in ("date", "datetime")

        for qualified in self.schema_context.get_columns():
            table, column = qualified.split(".", 1)
            info = self.schema_context.column_info(table, column)
            if info.get("is_sensitive"):
                continue

            score = 0.0
            signals = {}
            if is_num and info.get("is_numeric"):
                score = 0.45
                signals["type_compatibility"] = 0.45
            elif is_dt and info.get("is_date"):
                score = 0.50
                signals["type_compatibility"] = 0.50
            elif not is_num and not is_dt and info.get("is_text"):
                score = 0.40
                signals["type_compatibility"] = 0.40

            if score > 0.0:
                candidates.append({
                    "table": table,
                    "column": qualified,
                    "score": score,
                    "signals": signals,
                })
        return candidates

    def _is_fk_neighbor(self, t1: str, t2: str) -> bool:
        for fk in self.schema_context.foreign_keys:
            if (fk.get("child_table") == t1 and fk.get("parent_table") == t2) or \
               (fk.get("child_table") == t2 and fk.get("parent_table") == t1):
                return True
        return False

    def _is_reachable(self, t1: str, t2: str) -> bool:
        if t1 == t2:
            return True
        visited = {t1}
        queue = [t1]
        while queue:
            node = queue.pop(0)
            if node == t2:
                return True
            for fk in self.schema_context.foreign_keys:
                nbr = None
                if fk.get("child_table") == node:
                    nbr = fk.get("parent_table")
                elif fk.get("parent_table") == node:
                    nbr = fk.get("child_table")
                if nbr and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return False
