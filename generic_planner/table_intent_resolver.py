from __future__ import annotations

import re
from typing import Any

from .direct_queryir_builder import DirectQueryIRBuilder
from .join_policy import JoinPolicy
from .planner_result import GenericPlannerResult
from .schema_profile import SchemaProfile
from .schema_text_normalizer import column_name_variants, normalize_table_phrase


SHOW_WORDS = {"list", "show", "display", "view", "fetch", "get", "select"}
COUNT_PATTERNS = (r"\bcount\b", r"\bhow many\b", r"\bnumber of\b", r"\btotal\b")
ANALYTIC_BLOCKERS = (
    r"\btop\b",
    r"\bbottom\b",
    r"\baverage\b",
    r"\bavg\b",
    r"\bsum\b",
    r"\brevenue\b",
    r"\bsales\b",
    r"\bgroup\b",
    r"\btrend\b",
)
JOIN_LANGUAGE = (
    r"\bwith\s+\w+",
    r"\balong with\b",
    r"\band their\b",
    r"\bincluding\b",
    r"\bjoined with\b",
)
VALID_OPERATORS = {
    "equals",
    "not_equals",
    "contains",
    "in",
    "not_in",
    "greater_than",
    "greater_equal",
    "less_than",
    "less_equal",
}


class TableIntentResolver:
    def __init__(self, schema_profile: SchemaProfile):
        self.schema_profile = schema_profile
        self.builder = DirectQueryIRBuilder(schema_profile)

    def resolve(self, question: str) -> GenericPlannerResult:
        normalized = self._normalize_question(question)
        if not normalized:
            return GenericPlannerResult(False, reason="empty question")

        grouped_aggregation = bool(
            re.search(r"\bby\b", normalized)
            and re.search(r"\b(count|how many|number of|total|sum|average|avg|max|min)\b", normalized)
        )
        if grouped_aggregation or any(re.search(pattern, normalized) for pattern in ANALYTIC_BLOCKERS):
            return GenericPlannerResult(False, reason="analytical question requires model routing")

        table_result = self._single_table_match(question)
        if not table_result.get("ok"):
            table_result = self._disambiguate_with_filter(question, table_result)
        if not table_result.get("ok"):
            if any(re.search(pattern, normalized) for pattern in JOIN_LANGUAGE):
                return GenericPlannerResult(
                    handled=False,
                    reason="explicit join language requires normal routing",
                    debug={"join_policy": JoinPolicy.EXPLICIT_ONLY.value, "table_match": table_result},
                )
            return GenericPlannerResult(False, reason=table_result["reason"], debug=table_result)
        table = table_result["table"]

        filter_payload = self._extract_filter(question, table)
        if filter_payload.get("is_filter"):
            if not filter_payload.get("ok"):
                return GenericPlannerResult(False, reason=filter_payload["reason"], debug={"table_match": table_result, "filter": filter_payload})
            query_ir = self.builder.build_simple_filter(
                table=table,
                filter_column=filter_payload["column"],
                filter_operator=filter_payload["operator"],
                filter_value=filter_payload["value"],
                question=question,
                selected_columns=self._requested_projection_columns(question, table),
            )
            query_ir.metadata["generic_planner_debug"] = {
                "matched_table": table_result,
                "filter": filter_payload,
                "bypass_reason": "simple single-table filter",
                "join_policy": JoinPolicy.NONE.value,
            }
            return GenericPlannerResult(
                handled=True,
                intent="simple_filter",
                query_ir=query_ir,
                confidence=min(0.97, float(table_result["score"])),
                reason="simple single-table filter",
                debug=query_ir.metadata["generic_planner_debug"],
            )

        if any(re.search(pattern, normalized) for pattern in JOIN_LANGUAGE):
            return GenericPlannerResult(
                handled=False,
                reason="explicit join language requires normal routing",
                debug={"join_policy": JoinPolicy.EXPLICIT_ONLY.value, "table_match": table_result},
            )

        if self._is_count_question(normalized):
            query_ir = self.builder.build_count_records(table=table, question=question)
            query_ir.metadata["generic_planner_debug"] = {
                "matched_table": table_result,
                "bypass_reason": "simple table count",
                "join_policy": JoinPolicy.NONE.value,
            }
            return GenericPlannerResult(
                handled=True,
                intent="count_records",
                query_ir=query_ir,
                confidence=min(0.98, float(table_result["score"])),
                reason="simple table count",
                debug=query_ir.metadata["generic_planner_debug"],
            )

        if self._is_show_question(normalized):
            query_ir = self.builder.build_show_records(table=table, question=question)
            query_ir.metadata["generic_planner_debug"] = {
                "matched_table": table_result,
                "safe_selected_columns": query_ir.metadata.get("safe_selected_columns", []),
                "bypass_reason": "simple table listing",
                "join_policy": JoinPolicy.NONE.value,
            }
            return GenericPlannerResult(
                handled=True,
                intent="show_records",
                query_ir=query_ir,
                confidence=min(0.98, float(table_result["score"])),
                reason="simple table listing",
                debug=query_ir.metadata["generic_planner_debug"],
            )

        return GenericPlannerResult(False, reason="no simple table intent detected", debug={"table_match": table_result})

    def _single_table_match(self, question: str) -> dict[str, Any]:
        matches = self.schema_profile.find_table_matches(question)
        strong = [match for match in matches if match["score"] >= 0.80]
        if not strong:
            tables = self.schema_profile.get_tables()
            if len(tables) == 1:
                return {
                    "ok": True,
                    "table": tables[0],
                    "score": 0.90,
                    "matched_text": "only table in schema",
                    "match_type": "single_table_schema",
                }
            return {"ok": False, "reason": "no clear table match", "matches": matches[:3]}
        top = strong[0]
        if len(strong) > 1 and strong[1]["table"] != top["table"] and strong[1]["score"] >= top["score"] - 0.05:
            return {"ok": False, "reason": "ambiguous table match", "matches": strong[:3]}
        return {"ok": True, **top}

    def _disambiguate_with_filter(self, question: str, table_result: dict[str, Any]) -> dict[str, Any]:
        if table_result.get("reason") != "ambiguous table match":
            return table_result
        matches = table_result.get("matches") or []
        viable = []
        for match in matches:
            table = match.get("table")
            if not table:
                continue
            filter_payload = self._extract_filter(question, table)
            if filter_payload.get("ok"):
                viable.append((match, filter_payload))
        if len(viable) != 1:
            return table_result
        match, filter_payload = viable[0]
        return {
            "ok": True,
            **match,
            "match_type": f"{match.get('match_type')}_filter_disambiguated",
            "disambiguated_by_filter": filter_payload.get("column"),
        }

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", str(question or "").lower()).strip()

    @staticmethod
    def _is_count_question(normalized: str) -> bool:
        return any(re.search(pattern, normalized) for pattern in COUNT_PATTERNS)

    @staticmethod
    def _is_show_question(normalized: str) -> bool:
        first = normalized.split(" ", 1)[0] if normalized else ""
        if first in SHOW_WORDS:
            return True
        return any(re.search(rf"\b{word}\b", normalized) for word in SHOW_WORDS)

    def _extract_filter(self, question: str, table: str) -> dict[str, Any]:
        normalized = self._normalize_question(question)
        patterns = [
            r"\bwhere\s+(?P<column>[A-Za-z_][\w\s-]*?)\s+(?P<operator>on or after|on or before|greater than|less than|equals|contains|is not|not|is|=|!=|>=|<=|>|<)\s+(?P<value>.+)$",
            r"\bwith\s+(?P<column>[A-Za-z_][\w\s-]*?)\s+(?P<value>[A-Za-z0-9_.:-]+)$",
            r"\b(?P<column>created|updated|date|[A-Za-z_][\w-]*_date)\s+(?P<operator>after|before|on or after|on or before)\s+(?P<value>[A-Za-z0-9_.:-]+)$",
        ]
        payload: dict[str, Any] = {"is_filter": False}

        # Adjective status form: "show active users" / "list completed jobs".
        status_columns = [
            column["name"] for column in self.schema_profile.get_columns(table)
            if normalize_table_phrase(column["name"]) in {"status", "state", "active", "enabled"}
        ]
        table_phrases = sorted(self.schema_profile._table_variants.get(table, {table}), key=len, reverse=True)
        for table_phrase in table_phrases:
            phrase = normalize_table_phrase(table_phrase)
            match = re.search(rf"\b(?:show|list|display|get)\s+(?P<value>[a-z][\w-]*)\s+{re.escape(phrase)}\b", normalized)
            if match and status_columns and match.group("value") not in {"all", "the"}:
                return {
                    "is_filter": True,
                    "ok": True,
                    "column": status_columns[0],
                    "operator": "equals",
                    "value": self._clean_value(match.group("value")),
                    "column_match": {"table": table, "column": status_columns[0], "score": 0.95, "match_type": "status_adjective"},
                    "raw": match.group(0),
                }
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            groups = match.groupdict()
            column_text = (groups.get("column") or "").strip()
            value = self._clean_value(groups.get("value") or "")
            if match.group(0).startswith("with ") and value.lower() in {"name", "names", "detail", "details", "info", "information"}:
                return {"is_filter": False}
            operator = self._operator(groups.get("operator") or "equals")
            if operator not in VALID_OPERATORS or not value:
                return {"is_filter": True, "ok": False, "reason": "unsupported simple filter", "raw": match.group(0)}
            column_match = self._match_filter_column(column_text, table)
            if not column_match:
                return {
                    "is_filter": True,
                    "ok": False,
                    "reason": "filter column not found on matched base table",
                    "column_text": column_text,
                    "table": table,
                }
            return {
                "is_filter": True,
                "ok": True,
                "column": column_match["column"],
                "operator": operator,
                "value": value,
                "column_match": column_match,
                "raw": match.group(0),
            }

        # Lookup forms such as "episode written by Mark Tinker" and
        # "school whose season was in 2012".  These only activate when the
        # schema supplies a strong matching column, keeping the direct path
        # conservative on ambiguous schemas.
        connector_patterns = [
            r"\b(?P<column>written by|directed by|produced by|arranged by)\s+(?P<value>[^?.!]+)",
            r"\b(?P<column>titled|named|called)\s+[\"']?(?P<value>[^\"'?.!]+)",
        ]
        for pattern in connector_patterns:
            for match in re.finditer(pattern, normalized):
                column_text = match.group("column").strip()
                value = self._clean_value(match.group("value"))
                column_match = self._match_filter_column(column_text, table)
                if column_match and value and len(value.split()) <= 12:
                    return {
                        "is_filter": True,
                        "ok": True,
                        "column": column_match["column"],
                        "operator": "equals",
                        "value": value,
                        "column_match": column_match,
                        "raw": match.group(0),
                    }

        for column in self.schema_profile.get_columns(table):
            for variant in sorted(column_name_variants(column["name"]), key=len, reverse=True):
                phrase = normalize_table_phrase(variant)
                if not phrase:
                    continue
                match = re.search(
                    rf"\b{re.escape(phrase)}\b\s+(?:is|was|were|being|equals|=)\s+(?:in\s+)?(?P<value>[^?.!,]+)",
                    normalized,
                )
                if match:
                    value = self._clean_value(match.group("value"))
                    if value and len(value.split()) <= 12:
                        return {
                            "is_filter": True,
                            "ok": True,
                            "column": column["name"],
                            "operator": "equals",
                            "value": value,
                            "column_match": {"table": table, "column": column["name"], "score": 1.0, "match_type": "exact_column_phrase"},
                            "raw": match.group(0),
                        }

        # Short form: "users status active" or "status completed".
        if not self._is_show_question(normalized):
            for column in self.schema_profile.get_columns(table):
                variants = column_name_variants(column["name"])
                for variant in sorted(variants, key=len, reverse=True):
                    phrase = normalize_table_phrase(variant)
                    if not phrase:
                        continue
                    match = re.search(rf"\b{re.escape(phrase)}\s+(?P<value>[A-Za-z0-9_.:-]+)\b", normalized)
                    if match and not any(re.search(pattern, normalized) for pattern in COUNT_PATTERNS):
                        return {
                            "is_filter": True,
                            "ok": True,
                            "column": column["name"],
                            "operator": "equals",
                            "value": self._clean_value(match.group("value")),
                            "column_match": {"table": table, "column": column["name"], "score": 0.9, "match_type": "short_form"},
                            "raw": match.group(0),
                        }
        return payload

    def _requested_projection_columns(self, question: str, table: str) -> list[str] | None:
        normalized = self._normalize_question(question)
        patterns = [
            r"^(?:what|which)\s+(?:is|was|are|were)?\s*(?:the\s+)?(?P<target>.+?)\s+(?:of|for|with|where|whose|when|that|who)\b",
            r"^who\s+(?P<target>[a-z][\w\s/-]*?)\s+(?:the\s+)?(?:episode|track|player|team|record)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            matches = self.schema_profile.find_column_matches(match.group("target"), table=table)
            if matches and float(matches[0].get("score", 0.0)) >= 0.80:
                return [str(matches[0]["column"])]
        return None

    def _match_filter_column(self, column_text: str, table: str) -> dict[str, Any] | None:
        matches = self.schema_profile.find_column_matches(column_text, table=table)
        if not matches:
            return None
        top = matches[0]
        if top["score"] < 0.70:
            return None
        return top

    @staticmethod
    def _operator(raw_operator: str) -> str:
        op = str(raw_operator or "").strip().lower()
        return {
            "is": "equals",
            "equals": "equals",
            "=": "equals",
            "not": "not_equals",
            "is not": "not_equals",
            "!=": "not_equals",
            "after": "greater_than",
            ">": "greater_than",
            "greater than": "greater_than",
            "on or after": "greater_equal",
            ">=": "greater_equal",
            "before": "less_than",
            "<": "less_than",
            "less than": "less_than",
            "on or before": "less_equal",
            "<=": "less_equal",
            "contains": "contains",
        }.get(op, "equals")

    @staticmethod
    def _clean_value(value: str) -> str:
        return str(value or "").strip().strip("'\"`.,;")
