from __future__ import annotations

import pytest

from capabilities import SQLCapabilityExtractor


def _caps(sql: str) -> set[str]:
    return set(SQLCapabilityExtractor().extract(sql).required_capabilities)


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id", {"AGGREGATION", "GROUP_BY"}),
        ("SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id HAVING SUM(amount) > 10", {"HAVING"}),
        ("SELECT CASE WHEN amount > 10 THEN 'big' ELSE 'small' END FROM orders", {"CASE_EXPRESSION"}),
        ("SELECT (SELECT MAX(amount) FROM orders) AS max_amount", {"SCALAR_SUBQUERY", "AGGREGATION"}),
        ("SELECT id FROM customers WHERE id IN (SELECT customer_id FROM orders)", {"IN_SUBQUERY"}),
        ("SELECT id FROM customers WHERE EXISTS (SELECT 1 FROM orders)", {"EXISTS_SUBQUERY"}),
        ("SELECT t.customer_id FROM (SELECT customer_id FROM orders) AS t", {"DERIVED_TABLE"}),
        (
            "SELECT c.id FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)",
            {"EXISTS_SUBQUERY", "CORRELATED_SUBQUERY"},
        ),
        ("SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) FROM orders", {"WINDOW_ROW_NUMBER"}),
        ("SELECT RANK() OVER (ORDER BY amount DESC) FROM orders", {"WINDOW_RANK"}),
        ("SELECT LAG(amount) OVER (ORDER BY order_date) FROM orders", {"WINDOW_LAG"}),
        ("SELECT SUM(amount) OVER (PARTITION BY customer_id) FROM orders", {"WINDOW_AGGREGATE"}),
        ("SELECT id FROM a UNION ALL SELECT id FROM b", {"UNION_ALL"}),
        ("SELECT id FROM a UNION SELECT id FROM b", {"UNION"}),
        ("SELECT id FROM a INTERSECT SELECT id FROM b", {"INTERSECT"}),
        ("SELECT id FROM a EXCEPT SELECT id FROM b", {"EXCEPT"}),
    ],
)
def test_sql_capability_extractor_detects_required_capabilities(sql: str, expected: set[str]) -> None:
    assert expected.issubset(_caps(sql))


def test_sql_capability_extractor_detects_safety_labels() -> None:
    extractor = SQLCapabilityExtractor()
    assert extractor.extract("INSERT INTO orders(id) VALUES (1)").safety_labels == ["MUTATION_INSERT"]
    assert extractor.extract("UPDATE orders SET amount = 1").safety_labels == ["MUTATION_UPDATE"]
    assert extractor.extract("DELETE FROM orders WHERE id = 1").safety_labels == ["MUTATION_DELETE"]
    assert extractor.extract("CREATE TABLE t(id INT)").safety_labels == ["DDL_CREATE"]


def test_multiple_simultaneous_capabilities_and_policy_are_separate() -> None:
    annotation = SQLCapabilityExtractor().extract(
        "SELECT c.region, SUM(o.amount) AS revenue "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "WHERE c.region = 'west' OR c.region = 'east' "
        "GROUP BY c.region ORDER BY revenue DESC LIMIT 3"
    )

    assert {
        "AGGREGATION",
        "GROUP_BY",
        "ONE_HOP_JOIN",
        "OR_FILTER",
        "ORDER_BY",
        "LIMIT",
    }.issubset(set(annotation.required_capabilities))
    assert annotation.understood is True
    assert annotation.currently_supported is False
    assert annotation.unsupported_required_capabilities == ["OR_FILTER"]
