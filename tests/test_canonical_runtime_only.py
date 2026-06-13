from __future__ import annotations

import inspect

from inference.prediction_orchestrator import PredictionOrchestrator
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel


def test_canonical_runtime_uses_query_ir_pipeline() -> None:
    source = inspect.getsource(PredictionOrchestrator.predict)

    assert "OptionCToIRConverter" not in source
    assert hasattr(PredictionOrchestrator(), "ir_converter")
    assert hasattr(PredictionOrchestrator(), "ir_validator")
    assert hasattr(PredictionOrchestrator(), "sql_renderer")
    assert hasattr(PredictionOrchestrator(), "sql_validator")
    assert RetrievalNL2SQLModel.__name__ == "RetrievalNL2SQLModel"


def test_canonical_model_does_not_call_old_engine() -> None:
    source = inspect.getsource(RetrievalNL2SQLModel.predict)

    assert "NL2SQLEngine" not in source
    assert ".generate(" not in source
    assert "orchestrator.predict" in source
