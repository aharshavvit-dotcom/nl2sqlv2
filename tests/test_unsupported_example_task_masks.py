from __future__ import annotations

from capabilities import SQLCapabilityExtractor
from capabilities.contracts import annotation_to_unsupported_example


def test_unsupported_example_uses_auxiliary_masks_without_full_ir_loss() -> None:
    sql = "SELECT RANK() OVER (ORDER BY amount DESC) AS rnk FROM orders"
    extractor = SQLCapabilityExtractor()
    annotation = extractor.extract(sql, unsupported_reason="window_function")
    annotation = extractor.with_conversion_result(annotation, {"success": False, "unsupported_reason": "window_function"})
    example = annotation_to_unsupported_example(annotation, "window_function")

    assert "WINDOW_RANK" in example.capabilities
    assert example.task_masks.capability == 1
    assert example.task_masks.table == 1
    assert example.task_masks.window == 1
    assert example.task_masks.full_query_ir == 0
    assert example.partial_supervision.full_query_ir_supported is False
