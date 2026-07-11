from __future__ import annotations

from capabilities import Capability, SafetyLabel
from capabilities.taxonomy import SUPPORTED_QUERYIR_V1_CAPABILITIES, capability_names


def test_capability_taxonomy_is_multilabel_and_separates_safety() -> None:
    capabilities = {
        Capability.AGGREGATION,
        Capability.GROUP_BY,
        Capability.WINDOW_RANK,
        Capability.MULTI_HOP_JOIN,
        Capability.ORDER_BY,
    }
    safety_labels = {SafetyLabel.MUTATION_DELETE}

    assert len(capabilities) == 5
    assert SafetyLabel.MUTATION_DELETE not in capabilities
    assert Capability.WINDOW_RANK not in SUPPORTED_QUERYIR_V1_CAPABILITIES
    assert capability_names(capabilities) == [
        "AGGREGATION",
        "GROUP_BY",
        "MULTI_HOP_JOIN",
        "ORDER_BY",
        "WINDOW_RANK",
    ]
