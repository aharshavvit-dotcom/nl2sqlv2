"""Gold Comparator — compares predicted QueryIR/SQL against gold-standard labels.

Provides field-by-field comparison of QueryIR structures and normalized SQL
string comparison.  This is the foundation for error classification and
correction example generation in the self-improvement loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing a single predicted QueryIR against its gold."""

    example_id: str
    match_score: float = 0.0
    field_matches: dict[str, bool] = field(default_factory=dict)
    field_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    is_exact_match: bool = False
    is_partial_match: bool = False


@dataclass
class SQLComparisonResult:
    """Result of comparing predicted SQL against gold SQL."""

    normalized_match: bool = False
    structural_match: bool = False
    keyword_overlap: float = 0.0


@dataclass
class BatchComparisonReport:
    """Aggregated comparison report across a batch of examples."""

    total: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    failures: int = 0
    field_accuracy: dict[str, float] = field(default_factory=dict)
    per_example: list[ComparisonResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

QUERYIR_FIELDS = [
    "intent",
    "base_table",
    "dimensions",
    "metrics",
    "filters",
    "date_filters",
    "group_by",
    "joins",
    "order_by",
    "limit",
]

# Weights for the composite match score.  Intent and base_table carry the
# most weight because getting them wrong usually invalidates the entire query.
FIELD_WEIGHTS: dict[str, float] = {
    "intent": 0.20,
    "base_table": 0.20,
    "dimensions": 0.10,
    "metrics": 0.13,
    "filters": 0.10,
    "date_filters": 0.08,
    "group_by": 0.04,
    "joins": 0.07,
    "order_by": 0.05,
    "limit": 0.03,
}


class GoldComparator:
    """Compares predicted QueryIR / SQL against gold-standard labels."""

    def compare(
        self,
        predicted: dict[str, Any],
        gold: dict[str, Any],
        schema: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare prediction and gold labels across QueryIR, SQL, and optional execution.

        Execution success is only an additional signal. A row is correct only
        when structural QueryIR/SQL comparison also passes.
        """

        pred_ir = predicted.get("query_ir") or predicted.get("predicted_query_ir") or {}
        gold_ir = gold.get("query_ir") or gold.get("gold_query_ir") or {}
        query_ir = self.compare_query_ir(pred_ir, gold_ir, example_id=str(gold.get("example_id") or predicted.get("example_id") or ""))
        sql = self.compare_sql(predicted.get("sql") or predicted.get("predicted_sql"), gold.get("sql") or gold.get("gold_sql") or gold.get("source_sql"))
        structure = None
        try:
            from execution_eval.sql_structure_comparator import SQLStructureComparator

            structure = SQLStructureComparator().compare(
                predicted.get("sql") or predicted.get("predicted_sql") or "",
                gold.get("sql") or gold.get("gold_sql") or gold.get("source_sql") or "",
                schema or {},
                dialect=(schema or {}).get("dialect", "sqlite"),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            structure = {"structure_score": 0.0, "errors": [str(exc)], "warnings": []}

        execution_match = None
        if execution_result:
            execution_match = bool(
                execution_result.get("result_match")
                or execution_result.get("execution_match")
                or (execution_result.get("result_comparison") or {}).get("result_match")
            )
        correctness = bool(query_ir.is_exact_match and structure.get("structure_score", 0.0) >= 0.99)
        return {
            "example_id": query_ir.example_id,
            "gold_comparison_score": round((0.65 * query_ir.match_score) + (0.35 * float(structure.get("structure_score", 0.0))), 6),
            "query_ir": query_ir,
            "sql": sql,
            "structure": structure,
            "execution_match": execution_match,
            "correct": correctness,
            "execution_success_alone_correct": False,
        }

    # ------------------------------------------------------------------
    # QueryIR comparison
    # ------------------------------------------------------------------

    def compare_query_ir(
        self,
        predicted: dict[str, Any],
        gold: dict[str, Any],
        example_id: str = "",
    ) -> ComparisonResult:
        """Field-by-field comparison of predicted vs gold QueryIR."""

        field_matches: dict[str, bool] = {}
        field_details: dict[str, dict[str, Any]] = {}

        for f in QUERYIR_FIELDS:
            pred_val = predicted.get(f)
            gold_val = gold.get(f)
            match, details = self._compare_field(f, pred_val, gold_val)
            field_matches[f] = match
            field_details[f] = details

        match_score = sum(
            FIELD_WEIGHTS.get(f, 0.0) * (1.0 if field_matches[f] else 0.0)
            for f in QUERYIR_FIELDS
        )
        # Normalize so perfect match == 1.0
        total_weight = sum(FIELD_WEIGHTS.get(f, 0.0) for f in QUERYIR_FIELDS)
        if total_weight:
            match_score /= total_weight

        is_exact = all(field_matches.values())
        is_partial = match_score >= 0.5 and not is_exact

        return ComparisonResult(
            example_id=example_id,
            match_score=round(match_score, 6),
            field_matches=field_matches,
            field_details=field_details,
            is_exact_match=is_exact,
            is_partial_match=is_partial,
        )

    # ------------------------------------------------------------------
    # SQL comparison
    # ------------------------------------------------------------------

    def compare_sql(
        self,
        predicted_sql: str | None,
        gold_sql: str | None,
    ) -> SQLComparisonResult:
        """Structural SQL comparison using normalized forms."""

        if not predicted_sql or not gold_sql:
            return SQLComparisonResult()

        norm_pred = _normalize_sql(predicted_sql)
        norm_gold = _normalize_sql(gold_sql)
        normalized_match = norm_pred == norm_gold

        struct_pred = _structural_tokens(predicted_sql)
        struct_gold = _structural_tokens(gold_sql)
        structural_match = struct_pred == struct_gold

        kw_pred = _sql_keywords(predicted_sql)
        kw_gold = _sql_keywords(gold_sql)
        union = kw_pred | kw_gold
        keyword_overlap = len(kw_pred & kw_gold) / len(union) if union else 1.0

        return SQLComparisonResult(
            normalized_match=normalized_match,
            structural_match=structural_match,
            keyword_overlap=round(keyword_overlap, 4),
        )

    # ------------------------------------------------------------------
    # Batch comparison
    # ------------------------------------------------------------------

    def compare_batch(
        self,
        predictions: list[dict[str, Any]],
        gold_examples: list[dict[str, Any]],
    ) -> BatchComparisonReport:
        """Compare a list of predictions against gold examples.

        Both lists are matched by ``example_id``.  If a prediction has
        no matching gold example it is counted as a failure.
        """

        gold_map: dict[str, dict[str, Any]] = {
            str(ex.get("example_id", idx)): ex
            for idx, ex in enumerate(gold_examples)
        }

        results: list[ComparisonResult] = []
        exact = partial = failures = 0
        field_correct: dict[str, int] = {f: 0 for f in QUERYIR_FIELDS}
        field_total: dict[str, int] = {f: 0 for f in QUERYIR_FIELDS}

        for pred in predictions:
            eid = str(pred.get("example_id", ""))
            gold = gold_map.get(eid)
            if gold is None:
                failures += 1
                results.append(ComparisonResult(example_id=eid))
                continue

            gold_ir = gold.get("query_ir") or gold.get("gold_query_ir") or {}
            pred_ir = pred.get("predicted_query_ir") or pred.get("query_ir") or {}
            cr = self.compare_query_ir(pred_ir, gold_ir, example_id=eid)
            results.append(cr)

            if cr.is_exact_match:
                exact += 1
            elif cr.is_partial_match:
                partial += 1
            else:
                failures += 1

            for f in QUERYIR_FIELDS:
                field_total[f] += 1
                if cr.field_matches.get(f, False):
                    field_correct[f] += 1

        field_accuracy = {
            f: round(field_correct[f] / field_total[f], 4) if field_total[f] else 0.0
            for f in QUERYIR_FIELDS
        }

        return BatchComparisonReport(
            total=len(predictions),
            exact_matches=exact,
            partial_matches=partial,
            failures=failures,
            field_accuracy=field_accuracy,
            per_example=results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compare_field(
        self,
        field_name: str,
        predicted: Any,
        gold: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Return (match, detail_dict) for a single QueryIR field."""

        details: dict[str, Any] = {"predicted": predicted, "gold": gold}

        # Scalar fields
        if field_name in ("intent", "base_table", "limit"):
            match = _scalar_match(predicted, gold)
            details["match"] = match
            return match, details

        # List-of-dict fields
        if field_name in ("metrics", "dimensions", "filters", "date_filters", "joins", "order_by"):
            keys = _PROJECTION_KEYS.get(field_name, [])
            match, overlap = _list_match(predicted, gold, keys)
            details["match"] = match
            details["overlap"] = overlap
            return match, details

        # Fallback: direct equality
        match = predicted == gold
        details["match"] = match
        return match, details


# ---------------------------------------------------------------------------
# Projection keys for list-of-dict comparison (mirrors DatasetScaleEvaluator)
# ---------------------------------------------------------------------------

_PROJECTION_KEYS: dict[str, list[str]] = {
    "metrics": ["aggregation", "expression"],
    "dimensions": ["expression"],
    "filters": ["expression", "operator", "value"],
    "date_filters": ["date_expression", "filter_type", "start_date", "end_date", "date_grain"],
    "joins": ["condition"],
    "order_by": ["expression", "direction"],
}


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def _scalar_match(predicted: Any, gold: Any) -> bool:
    """Compare two scalar values, treating None/missing as equal."""
    if predicted is None and gold is None:
        return True
    if predicted is None or gold is None:
        return False
    return str(predicted).strip().lower() == str(gold).strip().lower()


def _list_match(
    predicted: Any,
    gold: Any,
    projection_keys: list[str],
) -> tuple[bool, float]:
    """Compare two list-of-dict fields using projection keys.

    Returns (exact_match, overlap_ratio).
    """
    pred_list = predicted if isinstance(predicted, list) else []
    gold_list = gold if isinstance(gold, list) else []

    if not pred_list and not gold_list:
        return True, 1.0

    pred_tuples = sorted(
        tuple(item.get(k) for k in projection_keys)
        for item in pred_list
        if isinstance(item, dict)
    )
    gold_tuples = sorted(
        tuple(item.get(k) for k in projection_keys)
        for item in gold_list
        if isinstance(item, dict)
    )

    exact = pred_tuples == gold_tuples

    pred_set = set(pred_tuples)
    gold_set = set(gold_tuples)
    union = pred_set | gold_set
    overlap = len(pred_set & gold_set) / len(union) if union else 1.0

    return exact, round(overlap, 4)


# ---------------------------------------------------------------------------
# SQL normalization helpers
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")
_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having",
    "limit", "join", "inner", "left", "right", "outer", "on",
    "and", "or", "not", "in", "between", "like", "ilike", "as",
    "asc", "desc", "distinct", "count", "sum", "avg", "min", "max",
    "case", "when", "then", "else", "end", "is", "null", "union",
    "all", "exists", "with", "cast", "coalesce", "date_trunc",
    "strftime", "offset",
}


def _normalize_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    return _WHITESPACE.sub(" ", sql.strip().rstrip(";").lower()).strip()


def _structural_tokens(sql: str) -> list[str]:
    """Extract SQL keyword skeleton (ignoring literal values)."""
    tokens = _normalize_sql(sql).split()
    return [t for t in tokens if t in _SQL_KEYWORDS or "." in t or t == "*"]


def _sql_keywords(sql: str) -> set[str]:
    """Extract set of SQL keywords present in the query."""
    tokens = _normalize_sql(sql).split()
    return {t for t in tokens if t in _SQL_KEYWORDS}
