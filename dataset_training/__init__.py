"""Dataset-scale generic training and evaluation utilities."""

from .dataset_registry import DatasetRegistry
from .split_manager import DatasetSplitManager
from .leakage_checker import DatasetLeakageChecker
from .ir_corpus_builder import GenericIRCorpusBuilder
from .corpus_quality import CorpusQualityAnalyzer
from .dataset_evaluator import DatasetScaleEvaluator
from .capability_registry import ActiveCapabilityRegistry, CapabilityEntry, CapabilityStatus
from .synthetic_generator import SyntheticQueryGenerator, SyntheticProvenance, SchemaDefinition

__all__ = [
    "ActiveCapabilityRegistry",
    "CapabilityEntry",
    "CapabilityStatus",
    "CorpusQualityAnalyzer",
    "DatasetLeakageChecker",
    "DatasetRegistry",
    "DatasetScaleEvaluator",
    "DatasetSplitManager",
    "GenericIRCorpusBuilder",
    "SchemaDefinition",
    "SyntheticProvenance",
    "SyntheticQueryGenerator",
]

