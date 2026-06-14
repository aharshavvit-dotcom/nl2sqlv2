# Legacy V1 Runtime

`nl2sql_v1/` is retained as reference and compatibility code for schema inspection,
retriever support, and older tests. It is not the active SQL generation runtime.

The canonical runtime is:

```text
RetrievalNL2SQLModel -> PredictionOrchestrator -> QueryIR -> IRToSQLRenderer -> SQLValidator
```

New runtime work should target `inference/`, `ir/`, `validation/`, and `execution/`.

