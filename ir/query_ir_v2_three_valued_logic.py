"""SQL three-valued logic for QueryIR v2 predicate analysis.

SQL uses three-valued logic: TRUE, FALSE, UNKNOWN.
Key difference from classical Boolean logic:
  - NOT UNKNOWN = UNKNOWN
  - TRUE OR UNKNOWN = TRUE
  - FALSE AND UNKNOWN = UNKNOWN
  - UNKNOWN OR UNKNOWN = UNKNOWN

Critical: Do NOT simplify `A OR NOT A` to TRUE because A may be UNKNOWN.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence


class TruthValue(Enum):
    """SQL three-valued truth value."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __bool__(self) -> bool:
        """Only TRUE is truthy in Python context."""
        return self is TruthValue.TRUE


def sql_not(v: TruthValue) -> TruthValue:
    """SQL NOT with three-valued logic."""
    if v is TruthValue.TRUE:
        return TruthValue.FALSE
    if v is TruthValue.FALSE:
        return TruthValue.TRUE
    return TruthValue.UNKNOWN  # NOT UNKNOWN = UNKNOWN


def sql_and(values: Sequence[TruthValue]) -> TruthValue:
    """SQL AND with three-valued logic.

    FALSE AND anything = FALSE
    TRUE AND TRUE = TRUE
    Otherwise UNKNOWN.
    """
    has_unknown = False
    for v in values:
        if v is TruthValue.FALSE:
            return TruthValue.FALSE
        if v is TruthValue.UNKNOWN:
            has_unknown = True
    return TruthValue.UNKNOWN if has_unknown else TruthValue.TRUE


def sql_or(values: Sequence[TruthValue]) -> TruthValue:
    """SQL OR with three-valued logic.

    TRUE OR anything = TRUE
    FALSE OR FALSE = FALSE
    Otherwise UNKNOWN.
    """
    has_unknown = False
    for v in values:
        if v is TruthValue.TRUE:
            return TruthValue.TRUE
        if v is TruthValue.UNKNOWN:
            has_unknown = True
    return TruthValue.UNKNOWN if has_unknown else TruthValue.FALSE


def is_null_result(v: TruthValue) -> TruthValue:
    """SQL IS NULL returns TRUE/FALSE, never UNKNOWN."""
    if v is TruthValue.UNKNOWN:
        return TruthValue.TRUE
    return TruthValue.FALSE


def is_not_null_result(v: TruthValue) -> TruthValue:
    """SQL IS NOT NULL returns TRUE/FALSE, never UNKNOWN."""
    if v is TruthValue.UNKNOWN:
        return TruthValue.FALSE
    return TruthValue.TRUE


# ── Safety assertions ──────────────────────────────────────────────────
# These ensure the predicate system does not apply two-valued simplifications.

def assert_no_excluded_middle(a: TruthValue) -> None:
    """Verify that A OR NOT A is not assumed TRUE.

    In SQL three-valued logic: UNKNOWN OR NOT UNKNOWN = UNKNOWN OR UNKNOWN = UNKNOWN.
    """
    result = sql_or([a, sql_not(a)])
    if a is TruthValue.UNKNOWN:
        assert result is TruthValue.UNKNOWN, (
            f"Law of excluded middle must not hold for UNKNOWN: got {result}"
        )


def assert_no_contradiction_elimination(a: TruthValue) -> None:
    """Verify that A AND NOT A is not assumed FALSE.

    In SQL: UNKNOWN AND NOT UNKNOWN = UNKNOWN AND UNKNOWN = UNKNOWN.
    """
    result = sql_and([a, sql_not(a)])
    if a is TruthValue.UNKNOWN:
        assert result is TruthValue.UNKNOWN, (
            f"Law of non-contradiction must not hold for UNKNOWN: got {result}"
        )
