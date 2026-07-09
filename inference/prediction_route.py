from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class RuntimeMode(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class PredictionRoute(str, Enum):
    DIRECT_PLANNER = "direct_planner"
    RETRIEVAL = "retrieval"
    NEURAL = "neural"
    CLARIFICATION = "clarification"
    ABSTENTION = "abstention"


class DiagnosticContext(BaseModel):
    forced_route: PredictionRoute | None = None
    cache_read_enabled: bool = False
    cache_write_enabled: bool = False
    telemetry_namespace: str = "offline_route_diagnostics"
    feedback_enabled: bool = False
    persist_runtime_state: bool = False
    runtime_mode: RuntimeMode | str = RuntimeMode.DEVELOPMENT


class DiagnosticRoutingNotAllowedError(RuntimeError):
    """Exception raised when forced routing is requested in production."""
    pass
