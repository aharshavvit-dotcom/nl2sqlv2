from __future__ import annotations

from capabilities import SQLCapabilityExtractor


def test_training_and_inference_capability_extraction_are_deterministic() -> None:
    sql = (
        "SELECT c.region, SUM(o.amount) AS revenue "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "WHERE c.region = 'west' "
        "GROUP BY c.region ORDER BY revenue DESC LIMIT 3"
    )
    extractor = SQLCapabilityExtractor()
    training_annotation = extractor.extract(
        sql,
        example_id="train-example",
        dataset_source="training",
        database_identifier="sales",
    )
    inference_annotation = extractor.extract(
        sql,
        example_id="runtime-example",
        dataset_source="runtime",
        database_identifier="sales",
    )

    assert training_annotation.required_capabilities == inference_annotation.required_capabilities
    assert training_annotation.partial_supervision.referenced_tables == inference_annotation.partial_supervision.referenced_tables
    assert training_annotation.partial_supervision.filter_columns == inference_annotation.partial_supervision.filter_columns
    assert training_annotation.partial_supervision.join_edges == inference_annotation.partial_supervision.join_edges
