"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import pytest

from inference.prediction_models import RetrievedCandidate
from inference.template_selector import TemplateSelector


def _candidate(template_id: str = "detail_rows") -> RetrievedCandidate:
    return RetrievedCandidate(
        rank=1,
        example_id="ex1",
        question="Show filtered rows",
        template_id=template_id,
        similarity_score=0.8,
        rerank_score=0.8,
    )


@pytest.mark.parametrize(
    ("question", "expected_template"),
    [
        ("Orders where status is completed", "simple_filter"),
        ("Show products in category electronics", "simple_filter"),
        ("Show products from category electronics", "simple_filter"),
        ("Show revenue where region is west", "metric_summary"),
        ("Show revenue by region", "metric_by_dimension"),
    ],
)
def test_filter_phrases_select_expected_templates(question: str, expected_template: str) -> None:
    selected = TemplateSelector().select_template([_candidate()], question)

    assert selected["template_id"] == expected_template

