"""NL2SQLService — canonical production API facade.

This is the single entry point for all NL-to-SQL inference.
Application code (Streamlit, FastAPI, CLI) should call this service
rather than importing from model-specific modules directly.

Usage::

    service = NL2SQLService.from_bundle("artifacts/model_bundle")
    result = service.predict("show total sales by region", schema=schema)
    print(result.sql)
    print(result.query_ir)
    print(result.confidence)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from db.schema_graph import SchemaGraph


@dataclass
class NL2SQLResult:
    """Unified prediction result from the NL2SQL system."""

    question: str
    sql: str
    query_ir: dict[str, Any]
    confidence: float
    confidence_breakdown: dict[str, float] = field(default_factory=dict)
    source_model: str = ""
    model_version: str = ""
    route: str = ""
    latency_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    is_safe: bool = True
    abstained: bool = False
    abstention_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return {
            "question": self.question,
            "sql": self.sql,
            "query_ir": self.query_ir,
            "confidence": self.confidence,
            "confidence_breakdown": self.confidence_breakdown,
            "source_model": self.source_model,
            "model_version": self.model_version,
            "route": self.route,
            "latency_ms": self.latency_ms,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "is_safe": self.is_safe,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }


class NL2SQLService:
    """Canonical NL-to-SQL inference service.

    Wraps the retrieval model, neural model, and adaptive router
    behind a single interface. All application code should use this
    class rather than importing from individual model modules.
    """

    def __init__(
        self,
        retrieval_model: Any = None,
        neural_predictor: Any = None,
        prediction_orchestrator: Any = None,
        schema: SchemaGraph | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._retrieval = retrieval_model
        self._neural = neural_predictor
        self._orchestrator = prediction_orchestrator
        self._schema = schema
        self._config = config or {}
        self._model_version = str(self._config.get("model_version", ""))
        self._abstention_threshold = float(self._config.get("abstention_threshold", 0.20))

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        schema: SchemaGraph | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> "NL2SQLService":
        """Load all models from a model bundle directory.

        Parameters
        ----------
        bundle_dir:
            Path to the model bundle (contains retrieval/, neural/, config.yaml).
        schema:
            Optional database schema for inference.
        config_overrides:
            Optional config overrides applied after loading bundle config.
        """
        bundle_path = Path(bundle_dir)
        config: dict[str, Any] = {}

        # Load bundle config
        config_path = bundle_path / "config.yaml"
        if config_path.exists():
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config.update(config_overrides or {})

        # Load retrieval model
        retrieval_model = None
        retrieval_dir = bundle_path / "retrieval"
        if retrieval_dir.exists():
            try:
                from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
                retrieval_model = RetrievalNL2SQLModel.load(str(retrieval_dir), schema=schema)
            except Exception as exc:
                import warnings
                warnings.warn(f"Could not load retrieval model: {exc}", stacklevel=2)

        # Load neural predictor
        neural_predictor = None
        neural_dir = bundle_path / "neural"
        if neural_dir.exists() and (neural_dir / "model.pt").exists():
            try:
                from neural_ir.predictor import NeuralIRPredictor
                neural_predictor = NeuralIRPredictor.load(str(neural_dir))
            except Exception as exc:
                import warnings
                warnings.warn(f"Could not load neural model: {exc}", stacklevel=2)

        # Load orchestrator
        orchestrator = None
        try:
            from inference.prediction_orchestrator import PredictionOrchestrator
            orchestrator = PredictionOrchestrator(
                retrieval_model=retrieval_model,
                neural_predictor=neural_predictor,
                config=config,
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"Could not initialize orchestrator: {exc}", stacklevel=2)

        return cls(
            retrieval_model=retrieval_model,
            neural_predictor=neural_predictor,
            prediction_orchestrator=orchestrator,
            schema=schema,
            config=config,
        )

    def predict(
        self,
        question: str,
        schema: SchemaGraph | dict[str, Any] | None = None,
        db_id: str | None = None,
        dialect: str = "sqlite",
        max_results: int = 1,
    ) -> NL2SQLResult:
        """Generate SQL from a natural language question.

        Parameters
        ----------
        question:
            Natural language query.
        schema:
            Database schema (overrides the service-level schema).
        db_id:
            Database identifier.
        dialect:
            SQL dialect (sqlite, postgres).
        max_results:
            Maximum number of candidate predictions.

        Returns
        -------
        NL2SQLResult with sql, query_ir, confidence, and diagnostics.
        """
        start = time.perf_counter()
        effective_schema = schema or self._schema
        warnings_list: list[str] = []

        try:
            if self._orchestrator is not None:
                raw_result = self._orchestrator.predict(
                    question=question,
                    schema=effective_schema,
                    db_id=db_id,
                    dialect=dialect,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                return self._from_orchestrator_result(question, raw_result, elapsed_ms)

            if self._retrieval is not None:
                raw_result = self._retrieval.predict(question, schema=effective_schema)
                elapsed_ms = (time.perf_counter() - start) * 1000
                return self._from_retrieval_result(question, raw_result, elapsed_ms)

            return NL2SQLResult(
                question=question,
                sql="",
                query_ir={},
                confidence=0.0,
                abstained=True,
                abstention_reason="No model loaded",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return NL2SQLResult(
                question=question,
                sql="",
                query_ir={},
                confidence=0.0,
                abstained=True,
                abstention_reason=f"Prediction error: {exc}",
                latency_ms=elapsed_ms,
                warnings=[str(exc)],
            )

    def _from_orchestrator_result(
        self, question: str, raw: dict[str, Any], elapsed_ms: float
    ) -> NL2SQLResult:
        prediction = raw.get("prediction") or {}
        confidence = float(prediction.get("confidence") or raw.get("confidence", 0.0))
        abstained = confidence < self._abstention_threshold
        return NL2SQLResult(
            question=question,
            sql=str(prediction.get("sql") or raw.get("sql", "")),
            query_ir=prediction.get("query_ir") or raw.get("query_ir") or {},
            confidence=confidence,
            confidence_breakdown=prediction.get("confidence_breakdown") or {},
            source_model=str(prediction.get("source_model") or raw.get("source_model", "")),
            model_version=self._model_version,
            route=str(raw.get("route", "")),
            latency_ms=elapsed_ms,
            diagnostics=raw.get("diagnostics") or {},
            is_safe=bool(raw.get("is_safe", True)),
            abstained=abstained,
            abstention_reason="Low confidence" if abstained else "",
        )

    def _from_retrieval_result(
        self, question: str, raw: Any, elapsed_ms: float
    ) -> NL2SQLResult:
        if hasattr(raw, "model_dump"):
            raw_dict = raw.model_dump()
        elif hasattr(raw, "dict"):
            raw_dict = raw.dict()
        else:
            raw_dict = dict(raw) if raw else {}

        confidence = float(raw_dict.get("confidence", 0.0))
        abstained = confidence < self._abstention_threshold
        return NL2SQLResult(
            question=question,
            sql=str(raw_dict.get("sql", "")),
            query_ir=raw_dict.get("query_ir") or {},
            confidence=confidence,
            source_model="retrieval",
            model_version=self._model_version,
            latency_ms=elapsed_ms,
            abstained=abstained,
            abstention_reason="Low confidence" if abstained else "",
        )

    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded and ready for inference."""
        return self._retrieval is not None or self._neural is not None

    @property
    def loaded_models(self) -> list[str]:
        """List of loaded model types."""
        models = []
        if self._retrieval is not None:
            models.append("retrieval")
        if self._neural is not None:
            models.append("neural")
        if self._orchestrator is not None:
            models.append("orchestrator")
        return models
