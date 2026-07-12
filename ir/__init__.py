from __future__ import annotations

from .ir_to_sql_renderer import IRToSQLRenderer
from .ir_roundtrip_validator import IRRoundtripValidator
from .ir_validator import IRValidator
from .option_c_to_ir import RetrievalIRConverter, OptionCToIRConverter
from .semantic_metric_resolver import SemanticMetricResolver
from .sql_to_ir_converter import SQLToIRConverter
from .sql_to_ir_errors import SQLToIRError, UnsupportedSQLPattern
from .query_ir_models import (
    IRDateFilter,
    IRDimension,
    IRExpression,
    IRFilter,
    IRJoin,
    IRMetric,
    IROrderBy,
    IRValidationIssue,
    IRValidationResult,
    QueryIR,
    diff_query_ir,
)
from .query_ir_migration import QueryIRCompatibilityError, convert_v2_to_v1, migrate_v1_to_v2
from .query_ir_v2_boolean_canonicalization import canonicalize_predicate
from .query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer, QueryIRV2RenderingError
from .query_ir_v2_models import QueryNode
from .query_ir_v2_renderer_adapter import QueryIRV2RendererAdapter
from .query_ir_v2_scope import QueryAnalysis, QueryScopeAnalyzer
from .query_ir_v2_serialization import dumps_query_ir_v2, fingerprint_query_ir_v2, loads_query_ir_v2
from .query_ir_v2_validation import QueryIRV2Validator
from .query_ir_version_loader import detect_query_ir_version, load_query_ir
from .sql_to_query_ir_v2 import SQLToQueryIRV2Converter, SQLToQueryIRV2Error

__all__ = [
    "IRDateFilter",
    "IRDimension",
    "IRExpression",
    "IRFilter",
    "IRJoin",
    "IRMetric",
    "IROrderBy",
    "IRRoundtripValidator",
    "IRToSQLRenderer",
    "IRValidationIssue",
    "IRValidationResult",
    "IRValidator",
    "RetrievalIRConverter",
    "OptionCToIRConverter",  # Deprecated alias
    "QueryAnalysis",
    "QueryIR",
    "QueryIRCompatibilityError",
    "QueryIRV2NativeRenderer",
    "QueryIRV2RendererAdapter",
    "QueryIRV2RenderingError",
    "QueryIRV2Validator",
    "QueryNode",
    "QueryScopeAnalyzer",
    "SQLToQueryIRV2Converter",
    "SQLToQueryIRV2Error",
    "canonicalize_predicate",
    "convert_v2_to_v1",
    "detect_query_ir_version",
    "diff_query_ir",
    "dumps_query_ir_v2",
    "fingerprint_query_ir_v2",
    "load_query_ir",
    "loads_query_ir_v2",
    "migrate_v1_to_v2",
    "SemanticMetricResolver",
    "SQLToIRConverter",
    "SQLToIRError",
    "UnsupportedSQLPattern",
]
