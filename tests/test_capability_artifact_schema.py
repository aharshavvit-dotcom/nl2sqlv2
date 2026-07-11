from __future__ import annotations

import pytest
from pydantic import ValidationError

from capabilities import CapabilityAnnotation, SQLCapabilityExtractor


def test_capability_artifact_schema_rejects_unknown_fields() -> None:
    payload = SQLCapabilityExtractor().extract("SELECT id FROM users").model_dump(mode="json")
    payload["unknown_field"] = "not allowed"

    with pytest.raises(ValidationError):
        CapabilityAnnotation.model_validate(payload)


def test_capability_artifact_schema_roundtrips_typed_annotation() -> None:
    payload = SQLCapabilityExtractor().extract(
        "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
        example_id="ex1",
        dataset_source="mock",
        database_identifier="db1",
        schema={"tables": {"orders": {"columns": {"customer_id": {}}}}},
    ).model_dump(mode="json")

    loaded = CapabilityAnnotation.model_validate(payload)

    assert loaded.example_id == "ex1"
    assert loaded.dataset_source == "mock"
    assert "AGGREGATION" in loaded.required_capabilities
    assert loaded.partial_supervision.group_by_columns == ["customer_id"]
