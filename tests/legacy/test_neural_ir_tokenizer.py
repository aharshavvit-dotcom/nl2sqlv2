"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from neural_ir.tokenizer import tokenize


def test_tokenizer_splits_question_on_punctuation_and_whitespace() -> None:
    assert tokenize("Top 5 customers by sales last month") == [
        "top",
        "5",
        "customers",
        "by",
        "sales",
        "last",
        "month",
    ]

