"""Capability taxonomy and partial-supervision utilities for NL-to-SQL."""

from .contracts import (
    CapabilityAnnotation,
    PartialSQLSupervision,
    SafetyExample,
    SupportedQueryIRExample,
    TaskMasks,
    UnsupportedExecutableExample,
)
from .evaluation import CapabilityEvaluator
from .reporting import CapabilityDatasetReporter
from .sql_capability_extractor import SQLCapabilityExtractor
from .taxonomy import (
    ALL_CAPABILITIES,
    ALL_SAFETY_LABELS,
    SUPPORTED_QUERYIR_V1_CAPABILITIES,
    Capability,
    SafetyLabel,
    capability_names,
    safety_label_names,
)

__all__ = [
    "ALL_CAPABILITIES",
    "ALL_SAFETY_LABELS",
    "SUPPORTED_QUERYIR_V1_CAPABILITIES",
    "Capability",
    "CapabilityAnnotation",
    "CapabilityDatasetReporter",
    "CapabilityEvaluator",
    "PartialSQLSupervision",
    "SafetyExample",
    "SafetyLabel",
    "SQLCapabilityExtractor",
    "SupportedQueryIRExample",
    "TaskMasks",
    "UnsupportedExecutableExample",
    "capability_names",
    "safety_label_names",
]
