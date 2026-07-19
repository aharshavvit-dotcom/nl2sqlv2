# Canonical Owner Matrix

| Area | Classification | Canonical owner | Cleanup action |
| --- | --- | --- | --- |
| retrieval vs retriever | CONSOLIDATE | retrieval/ for index/reranker infrastructure; retriever/ currently retains runtime RetrievalNL2SQLModel wrapper | Plan import migration before deleting retriever/. |
| training vs training_ir | REVIEW_REQUIRED | training/train_model.py for integrated pipeline | Keep until commands are replaced or archived with tests/docs updated. |
| dataset_training vs datasets | KEEP_BOTH | datasets/ adapters; dataset_training/ corpus/split/leakage builders | No merge without API design. |
| models vs model_bundle | KEEP_BOTH | model_bundle/ for production bundles; models/ only ignored local artifact placeholder | Keep models/.gitkeep only. |
| evaluation reports in docs/reports vs artifacts/ | ARCHIVE | artifacts/pipeline/runs/<run_id>/reports for run-scoped generated reports | Move after review; no automatic deletion in this pass. |
