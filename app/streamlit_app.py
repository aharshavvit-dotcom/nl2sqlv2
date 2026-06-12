from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nl2sql_v1.executor import execute_select
from nl2sql_v1.feedback import append_feedback
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.verify_datasets import verify_all
from training.train_retriever_from_datasets import train_from_datasets


DEFAULT_DB = ROOT / "data" / "sample_retail.db"
EXAMPLES_PATH = ROOT / "training_data" / "examples.jsonl"
TEMPLATES_PATH = ROOT / "data" / "templates.yaml"
SYNONYMS_PATH = ROOT / "data" / "synonyms.yaml"
MODEL_PATH = ROOT / "models" / "tfidf_retriever.joblib"
ARTIFACT_DIR = ROOT / "artifacts" / "option_c_model"
FEEDBACK_PATH = ROOT / "feedback" / "feedback.jsonl"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_ready() -> bool:
    return (
        (ARTIFACT_DIR / "training_examples.jsonl").exists()
        and (ARTIFACT_DIR / "tfidf_vectorizer.pkl").exists()
        and (ARTIFACT_DIR / "tfidf_matrix.pkl").exists()
    )


def _load_model() -> RetrievalNL2SQLModel:
    return RetrievalNL2SQLModel.load(
        artifact_dir=ARTIFACT_DIR,
        sample_model_path=MODEL_PATH,
        sample_examples_path=EXAMPLES_PATH,
        templates_path=TEMPLATES_PATH,
        synonyms_path=SYNONYMS_PATH,
    )


def _dataset_missing_messages(selected: list[str]) -> list[str]:
    rows = {row.dataset: row for row in verify_all()}
    messages = []
    if "wikisql" in selected and not rows["WikiSQL"].ready:
        messages.append("WikiSQL missing. Run python scripts/download_datasets.py --datasets wikisql")
    if "spider" in selected and not rows["Spider"].ready:
        messages.append("Spider missing. Run python scripts/download_datasets.py --datasets spider")
    if "bird-mini" in selected and not rows["BIRD Mini-Dev"].ready:
        messages.append("BIRD Mini-Dev missing. Run python scripts/download_datasets.py --datasets bird-mini")
    return messages


st.set_page_config(page_title="Local NL-to-SQL V1", layout="wide")
st.title("Local Retrieval NL-to-SQL V1")

with st.expander("Model Status", expanded=True):
    training_report = _load_json(ARTIFACT_DIR / "training_report.json")
    evaluation_report = _load_json(ARTIFACT_DIR / "evaluation_report.json")
    dataset_stats = _load_json(ARTIFACT_DIR / "dataset_stats.json")
    status_cols = st.columns(4)
    status_cols[0].metric("Artifact", "large" if _artifact_ready() else "sample")
    status_cols[1].metric("Training examples", training_report.get("supported_examples", "sample"))
    status_cols[2].metric("Unsupported", dataset_stats.get("unsupported_examples", 0))
    status_cols[3].metric("Top-5 accuracy", f"{evaluation_report.get('top_5_template_accuracy', 0):.3f}")
    st.caption(f"Artifact path: {ARTIFACT_DIR}")
    st.write(
        {
            "datasets_used": training_report.get("datasets_used", []),
            "templates_covered": list((training_report.get("by_template") or {}).keys()),
            "supported_examples": training_report.get("supported_examples"),
            "train_examples": training_report.get("train_examples"),
            "validation_examples": training_report.get("validation_examples"),
            "test_examples": training_report.get("test_examples"),
            "unsupported_examples": dataset_stats.get("unsupported_examples"),
            "top_1_template_accuracy": evaluation_report.get("top_1_template_accuracy"),
            "top_5_template_accuracy": evaluation_report.get("top_5_template_accuracy"),
            "training_date": training_report.get("training_date"),
        }
    )

with st.expander("Dataset Training", expanded=False):
    c1, c2, c3 = st.columns(3)
    use_wikisql = c1.checkbox("WikiSQL", value=True)
    use_spider = c2.checkbox("Spider", value=True)
    use_bird = c3.checkbox("BIRD Mini-Dev", value=True)
    max_examples = st.number_input("Max examples per dataset (0 = no limit)", min_value=0, value=0, step=100)
    include_schema_text = st.checkbox("Include schema text")
    if st.button("Train From Local Datasets"):
        selected = []
        if use_wikisql:
            selected.append("wikisql")
        if use_spider:
            selected.append("spider")
        if use_bird:
            selected.append("bird-mini")
        if not selected:
            st.warning("Select at least one dataset.")
        else:
            missing_messages = _dataset_missing_messages(selected)
            if missing_messages:
                for message in missing_messages:
                    st.warning(message)
            else:
                with st.spinner("Training TF-IDF retriever from local datasets..."):
                    report = train_from_datasets(
                        selected,
                        artifact_dir=ARTIFACT_DIR,
                        max_examples=int(max_examples) or None,
                        include_schema_text=include_schema_text,
                    )
                st.success("Training complete.")
                st.json(report)

db_path_text = st.text_input("SQLite database path", value=str(DEFAULT_DB))
db_path = Path(db_path_text).expanduser()

left, right = st.columns([0.55, 0.45], vertical_alignment="top")

with left:
    question = st.text_input("Ask a question", value="Top 5 customers by sales")
    generate = st.button("Generate SQL", type="primary")

with right:
    st.caption("Local only: TF-IDF retrieval, YAML templates, RapidFuzz matching, SQLGlot validation.")

if not db_path.exists():
    st.warning(f"Database not found: {db_path}")
    st.stop()

schema = read_sqlite_schema(db_path)
with st.expander("Schema", expanded=True):
    for table in schema.tables.values():
        cols = ", ".join(f"{col.name} ({col.type})" for col in table.columns.values())
        st.markdown(f"**{table.name}**: {cols}")

if generate and question.strip():
    model = _load_model()
    try:
        result = model.predict(question, schema)
    except Exception as exc:
        st.error(f"Could not generate SQL for this schema: {exc}")
        st.stop()

    st.subheader("Retrieved Examples")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "rank": item.get("rank"),
                    "id": item.get("example_id"),
                    "similarity": round(float(item.get("similarity_score") or 0), 4),
                    "rerank": round(float(item.get("rerank_score") or 0), 4),
                    "question": item.get("question"),
                    "template": item.get("template_id"),
                }
                for item in result.retrieved_candidates
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Confidence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{result.confidence:.3f}")
    c2.metric("Tier", result.confidence_tier)
    c3.metric("Intent", result.intent or "unknown")

    st.subheader("Slots")
    slot_rows = [
        {
            "slot": name,
            "value": slot.get("value"),
            "source": slot.get("source"),
            "confidence": round(float(slot.get("confidence") or 0), 3),
        }
        for name, slot in result.slots.items()
    ]
    st.dataframe(pd.DataFrame(slot_rows), use_container_width=True, hide_index=True)

    st.subheader("Schema Mapping")
    mapping = result.schema_mapping
    mapping_rows = [
        {"item": "metric", "table": mapping.get("metric_table"), "column": mapping.get("metric_column"), "score": (mapping.get("match_scores") or {}).get("metric")},
        {"item": "dimension", "table": mapping.get("dimension_table"), "column": mapping.get("dimension_column"), "score": (mapping.get("match_scores") or {}).get("dimension")},
        {"item": "entity", "table": mapping.get("entity_table"), "column": None, "score": (mapping.get("match_scores") or {}).get("entity")},
        {"item": "date", "table": mapping.get("date_table"), "column": mapping.get("date_column"), "score": (mapping.get("match_scores") or {}).get("date")},
    ]
    st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)

    st.subheader("Join Plan")
    st.code((result.join_plan or {}).get("join_clause") or "(no joins needed)", language="sql")

    st.subheader("Generated SQL")
    st.code(result.sql or "", language="sql")

    c1, c2, c3 = st.columns(3)
    c1.metric("Template", result.template_id or "unknown")
    c2.metric("Validation", "passed" if result.validation.get("ok") else "failed")
    c3.metric("Selected example", (result.selected_candidate or {}).get("example_id", "none"))

    st.subheader("SQL Validation")
    checks = result.validation.get("checks") or {}
    st.dataframe(
        pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]),
        use_container_width=True,
        hide_index=True,
    )

    if result.validation.get("ok"):
        st.success("SQL validation passed.")
    else:
        st.error(result.validation.get("message", "SQL validation failed"))

    if result.warnings:
        st.subheader("Warnings")
        for warning in result.warnings:
            st.warning(warning)
    if result.clarification_questions:
        st.subheader("Clarification Questions")
        for clarification in result.clarification_questions:
            st.info(clarification)

    run_query = st.button("Run query", disabled=not result.validation.get("ok"))
    if run_query and result.validation.get("ok") and result.sql:
        try:
            df = execute_select(db_path, result.sql)
            st.subheader("Result DataFrame")
            st.dataframe(df, use_container_width=True)
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(str(exc))

    st.subheader("Feedback")
    rating = st.radio("Was this useful?", ["skip", "thumbs_up", "thumbs_down"], horizontal=True)
    notes = st.text_area("Notes", height=80)
    if st.button("Save feedback"):
        append_feedback(
            FEEDBACK_PATH,
            {
                "question": question,
                "sql": result.sql,
                "rating": rating,
                "notes": notes,
                "db_path": str(db_path),
                "retrieved_examples": [item.get("example_id") for item in result.retrieved_candidates],
            },
        )
        st.success("Feedback saved.")

    if st.checkbox("Show debug details"):
        st.json(result.model_dump())
elif not generate:
    st.info("Connect a SQLite database, ask a question, then generate SQL.")

if st.checkbox("Show sample table preview"):
    table_name = st.selectbox("Table", sorted(schema.tables))
    preview_sql = f"SELECT * FROM {table_name} LIMIT 20"
    df = execute_select(db_path, preview_sql)
    st.dataframe(pd.DataFrame(df), use_container_width=True)
