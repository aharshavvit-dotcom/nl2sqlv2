"""Legacy runtime/reference helpers.

The active application, evaluator, and tests use:
RetrievalNL2SQLModel -> PredictionOrchestrator -> QueryIR -> SQLValidator.
"""

__all__ = [
    "engine",
    "executor",
    "feedback",
    "join_resolver",
    "renderer",
    "retriever",
    "schema",
    "schema_matcher",
    "slot_extractor",
    "template_adapter",
    "validator",
]
