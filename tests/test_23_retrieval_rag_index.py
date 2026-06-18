from __future__ import annotations

from pathlib import Path

from retrieval import ExampleIndex, LocalRAGRetriever, PatternIndex, RetrievalReranker, SchemaIndex
from retrieval.rag_index_builder import RAGIndexBuilder
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel


def _examples() -> list[dict]:
    return [
        {
            "example_id": "show_users",
            "question": "list all users",
            "intent": "show_records",
            "template_id": "show_records",
            "schema": {"tables": {"users": {"columns": {"id": {}, "name": {}}}}},
            "query_ir": {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": []},
        },
        {
            "example_id": "top_sales",
            "question": "top customers by sales",
            "intent": "top_n_metric_by_dimension",
            "template_id": "top_n_metric_by_dimension",
            "schema": {"tables": {"customers": {"columns": {"customer_name": {}}}, "orders": {"columns": {"amount": {}}}}},
            "query_ir": {"intent": "top_n_metric_by_dimension", "template_id": "top_n_metric_by_dimension", "base_table": "orders", "required_tables": ["orders", "customers"], "joins": [{"condition": "orders.customer_id = customers.customer_id"}], "metrics": [{"expression": "orders.amount"}]},
        },
    ]


def test_rag_retriever_prioritizes_show_records_for_simple_listing() -> None:
    examples = _examples()
    example_index = ExampleIndex()
    schema_index = SchemaIndex()
    pattern_index = PatternIndex()
    example_index.build(examples)
    schema_index.build(examples)
    pattern_index.build(examples)
    retriever = LocalRAGRetriever(example_index, schema_index, pattern_index, RetrievalReranker())

    result = retriever.retrieve(
        "list all users",
        {"tables": {"users": {"columns": {"id": {}, "name": {}}}, "assignments": {"columns": {"id": {}}}}},
        top_k=2,
    )

    assert result["patterns"][0]["pattern"] == "show_records"
    assert result["reranked"][0]["example_id"] == "show_users"
    assert result["reranked"][0]["intent"] != "top_n_metric_by_dimension"


def test_analytics_question_retrieves_analytics_example() -> None:
    examples = _examples()
    example_index = ExampleIndex()
    schema_index = SchemaIndex()
    pattern_index = PatternIndex()
    example_index.build(examples)
    schema_index.build(examples)
    pattern_index.build(examples)

    result = LocalRAGRetriever(example_index, schema_index, pattern_index, RetrievalReranker()).retrieve(
        "top customers by sales",
        {"tables": {"customers": {"columns": {"customer_name": {}}}, "orders": {"columns": {"amount": {}}}}},
    )

    assert result["reranked"][0]["example_id"] == "top_sales"


def test_runtime_loader_prefers_rag_index_when_present(tmp_path: Path) -> None:
    RAGIndexBuilder().build(_examples(), tmp_path)

    model = RetrievalNL2SQLModel.load(artifact_dir=tmp_path)
    results = model.retriever.query_with_schema(
        "list all users",
        {"tables": {"users": {"columns": {"id": {}, "name": {}}}}},
        top_k=1,
    )

    assert model.metadata["retrieval_backend"] == "local_rag"
    assert results[0].example_id == "show_users"
