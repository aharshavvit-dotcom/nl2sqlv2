"""Grammar State Machine — Gate 3 Architecture.

Constrains the decoder to only produce syntactically valid QueryIR structures.
Uses a finite-state approach where each decoder step has a set of valid
next-token categories, preventing impossible transitions.

Behind `enable_grammar_decoder` feature flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GrammarState(str, Enum):
    """States in the QueryIR production grammar."""
    START = "START"
    SELECT = "SELECT"
    FROM = "FROM"
    JOIN = "JOIN"
    WHERE = "WHERE"
    GROUP_BY = "GROUP_BY"
    HAVING = "HAVING"
    ORDER_BY = "ORDER_BY"
    LIMIT = "LIMIT"
    CTE = "CTE"
    SUBQUERY = "SUBQUERY"
    SET_OP = "SET_OP"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class TokenCategory(str, Enum):
    """Categories of output tokens the decoder can produce."""
    TABLE_NAME = "TABLE_NAME"
    COLUMN_NAME = "COLUMN_NAME"
    AGGREGATION_FUNC = "AGGREGATION_FUNC"
    COMPARISON_OP = "COMPARISON_OP"
    BOOLEAN_OP = "BOOLEAN_OP"
    LITERAL_VALUE = "LITERAL_VALUE"
    JOIN_TYPE = "JOIN_TYPE"
    DIRECTION = "DIRECTION"
    KEYWORD = "KEYWORD"
    PUNCTUATION = "PUNCTUATION"
    ALIAS = "ALIAS"
    WINDOW_FUNC = "WINDOW_FUNC"
    SET_OP_TYPE = "SET_OP_TYPE"


# Transition table: state -> set of valid next states
_TRANSITIONS: dict[GrammarState, set[GrammarState]] = {
    GrammarState.START: {GrammarState.CTE, GrammarState.SELECT},
    GrammarState.CTE: {GrammarState.CTE, GrammarState.SELECT},
    GrammarState.SELECT: {GrammarState.FROM},
    GrammarState.FROM: {
        GrammarState.JOIN, GrammarState.WHERE,
        GrammarState.GROUP_BY, GrammarState.ORDER_BY,
        GrammarState.LIMIT, GrammarState.SET_OP,
        GrammarState.COMPLETE,
    },
    GrammarState.JOIN: {
        GrammarState.JOIN, GrammarState.WHERE,
        GrammarState.GROUP_BY, GrammarState.ORDER_BY,
        GrammarState.LIMIT, GrammarState.SET_OP,
        GrammarState.COMPLETE,
    },
    GrammarState.WHERE: {
        GrammarState.GROUP_BY, GrammarState.ORDER_BY,
        GrammarState.LIMIT, GrammarState.SET_OP,
        GrammarState.COMPLETE,
    },
    GrammarState.GROUP_BY: {
        GrammarState.HAVING, GrammarState.ORDER_BY,
        GrammarState.LIMIT, GrammarState.SET_OP,
        GrammarState.COMPLETE,
    },
    GrammarState.HAVING: {
        GrammarState.ORDER_BY, GrammarState.LIMIT,
        GrammarState.SET_OP, GrammarState.COMPLETE,
    },
    GrammarState.ORDER_BY: {
        GrammarState.LIMIT, GrammarState.SET_OP,
        GrammarState.COMPLETE,
    },
    GrammarState.LIMIT: {
        GrammarState.SET_OP, GrammarState.COMPLETE,
    },
    GrammarState.SET_OP: {GrammarState.SELECT},
    GrammarState.SUBQUERY: {GrammarState.SELECT},
    GrammarState.COMPLETE: set(),
    GrammarState.ERROR: set(),
}

# Valid token categories per state
_STATE_TOKENS: dict[GrammarState, set[TokenCategory]] = {
    GrammarState.START: {TokenCategory.KEYWORD},
    GrammarState.CTE: {TokenCategory.TABLE_NAME, TokenCategory.KEYWORD, TokenCategory.ALIAS},
    GrammarState.SELECT: {
        TokenCategory.COLUMN_NAME, TokenCategory.TABLE_NAME,
        TokenCategory.AGGREGATION_FUNC, TokenCategory.WINDOW_FUNC,
        TokenCategory.ALIAS, TokenCategory.PUNCTUATION,
        TokenCategory.LITERAL_VALUE, TokenCategory.KEYWORD,
    },
    GrammarState.FROM: {TokenCategory.TABLE_NAME, TokenCategory.ALIAS, TokenCategory.KEYWORD},
    GrammarState.JOIN: {
        TokenCategory.TABLE_NAME, TokenCategory.JOIN_TYPE,
        TokenCategory.COLUMN_NAME, TokenCategory.COMPARISON_OP,
        TokenCategory.ALIAS, TokenCategory.KEYWORD,
    },
    GrammarState.WHERE: {
        TokenCategory.COLUMN_NAME, TokenCategory.TABLE_NAME,
        TokenCategory.COMPARISON_OP, TokenCategory.BOOLEAN_OP,
        TokenCategory.LITERAL_VALUE, TokenCategory.KEYWORD,
        TokenCategory.PUNCTUATION,
    },
    GrammarState.GROUP_BY: {
        TokenCategory.COLUMN_NAME, TokenCategory.TABLE_NAME,
        TokenCategory.PUNCTUATION,
    },
    GrammarState.HAVING: {
        TokenCategory.AGGREGATION_FUNC, TokenCategory.COLUMN_NAME,
        TokenCategory.COMPARISON_OP, TokenCategory.BOOLEAN_OP,
        TokenCategory.LITERAL_VALUE, TokenCategory.KEYWORD,
    },
    GrammarState.ORDER_BY: {
        TokenCategory.COLUMN_NAME, TokenCategory.TABLE_NAME,
        TokenCategory.DIRECTION, TokenCategory.PUNCTUATION,
        TokenCategory.AGGREGATION_FUNC,
    },
    GrammarState.LIMIT: {TokenCategory.LITERAL_VALUE},
    GrammarState.SET_OP: {TokenCategory.SET_OP_TYPE, TokenCategory.KEYWORD},
}


@dataclass
class GrammarTransition:
    """Record of a state machine transition."""
    from_state: GrammarState
    to_state: GrammarState
    token_category: TokenCategory | None = None
    is_valid: bool = True


class GrammarStateMachine:
    """Finite-state machine for constraining QueryIR production.

    Usage:
        fsm = GrammarStateMachine()
        valid_next = fsm.valid_transitions()  # What can come next?
        fsm.transition(GrammarState.SELECT)   # Move to SELECT
        mask = fsm.token_mask()               # Which token categories are valid?
    """

    def __init__(self) -> None:
        self._state = GrammarState.START
        self._history: list[GrammarTransition] = []
        self._depth = 0  # Subquery/CTE nesting depth
        self._max_depth = 4

    @property
    def state(self) -> GrammarState:
        return self._state

    @property
    def is_complete(self) -> bool:
        return self._state == GrammarState.COMPLETE

    @property
    def is_error(self) -> bool:
        return self._state == GrammarState.ERROR

    def valid_transitions(self) -> set[GrammarState]:
        """Return the set of valid next states."""
        valid = _TRANSITIONS.get(self._state, set())
        # Don't allow deeper nesting than max
        if self._depth >= self._max_depth:
            valid = valid - {GrammarState.SUBQUERY, GrammarState.CTE}
        return valid

    def can_transition(self, target: GrammarState) -> bool:
        return target in self.valid_transitions()

    def transition(self, target: GrammarState, token_category: TokenCategory | None = None) -> bool:
        """Attempt to transition to target state. Returns True if valid."""
        is_valid = self.can_transition(target)
        self._history.append(GrammarTransition(
            from_state=self._state,
            to_state=target,
            token_category=token_category,
            is_valid=is_valid,
        ))
        if is_valid:
            if target in {GrammarState.SUBQUERY, GrammarState.CTE}:
                self._depth += 1
            self._state = target
        else:
            self._state = GrammarState.ERROR
        return is_valid

    def token_mask(self) -> set[TokenCategory]:
        """Return valid token categories for the current state."""
        return _STATE_TOKENS.get(self._state, set())

    def token_mask_vector(self, all_categories: list[TokenCategory]) -> list[bool]:
        """Return a boolean mask vector over all token categories."""
        valid = self.token_mask()
        return [cat in valid for cat in all_categories]

    @property
    def history(self) -> list[GrammarTransition]:
        return list(self._history)

    def reset(self) -> None:
        self._state = GrammarState.START
        self._history.clear()
        self._depth = 0

    def validate_sequence(self, states: list[GrammarState]) -> tuple[bool, int]:
        """Validate a full state sequence. Returns (valid, failing_index)."""
        self.reset()
        for i, state in enumerate(states):
            if not self.transition(state):
                return False, i
        return True, -1


__all__ = [
    "GrammarState",
    "GrammarStateMachine",
    "GrammarTransition",
    "TokenCategory",
]
