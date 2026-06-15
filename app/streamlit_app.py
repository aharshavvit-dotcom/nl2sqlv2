from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.query_executor import execute_select
from app.safe_preview import build_safe_preview_sql
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
OPTION_A_ARTIFACT_DIR = ROOT / "artifacts" / "option_a_ir_model"
OPTION_A_V2_ARTIFACT_DIR = ROOT / "artifacts" / "option_a_ir_model_v2"
FEEDBACK_PATH = ROOT / "feedback" / "feedback.jsonl"
EVALUATION_DIR = ROOT / "evaluation"
GOLDEN_RESULTS_PATH = EVALUATION_DIR / "golden_runtime_report.json"


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


def _option_a_artifact_dir() -> Path:
    if _option_a_ready(OPTION_A_V2_ARTIFACT_DIR):
        return OPTION_A_V2_ARTIFACT_DIR
    if _option_a_ready(OPTION_A_ARTIFACT_DIR):
        return OPTION_A_ARTIFACT_DIR
    return OPTION_A_V2_ARTIFACT_DIR


def _option_a_ready(path: Path | None = None) -> bool:
    artifact_dir = path or _option_a_artifact_dir()
    return (
        (artifact_dir / "model.pt").exists()
        and (artifact_dir / "vocab.json").exists()
        and (artifact_dir / "label_maps.json").exists()
        and (artifact_dir / "config.yaml").exists()
    )


def _option_a_version() -> str:
    if _option_a_ready(OPTION_A_V2_ARTIFACT_DIR):
        return "v2"
    if not _option_a_ready(OPTION_A_ARTIFACT_DIR):
        return "none"
    config_text = (OPTION_A_ARTIFACT_DIR / "config.yaml").read_text(encoding="utf-8")
    if "v1_5" in config_text or "v1.5" in config_text:
        return "v1.5"
    return "v1"


def _load_model() -> RetrievalNL2SQLModel:
    return RetrievalNL2SQLModel.load(
        artifact_dir=ARTIFACT_DIR,
        sample_model_path=MODEL_PATH,
        sample_examples_path=EXAMPLES_PATH,
        templates_path=TEMPLATES_PATH,
        synonyms_path=SYNONYMS_PATH,
        option_a_model_dir=_option_a_artifact_dir() if _option_a_ready() else None,
    )


def _golden_results() -> dict[str, Any]:
    return _load_json(GOLDEN_RESULTS_PATH)


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


st.set_page_config(page_title="Local QueryIR NL-to-SQL", layout="wide")
st.title("Local QueryIR NL-to-SQL")

with st.expander("Model Status", expanded=True):
    evaluation_path = ARTIFACT_DIR / "evaluation_report.json"
    training_report = _load_json(ARTIFACT_DIR / "training_report.json")
    evaluation_report = _load_json(evaluation_path)
    dataset_stats = _load_json(ARTIFACT_DIR / "dataset_stats.json")
    option_c_ready = _artifact_ready()
    option_a_ready = _option_a_ready()
    option_a_version = _option_a_version()
    status_cols = st.columns(6)
    status_cols[0].metric("Option C artifact", "large" if option_c_ready else "sample")
    status_cols[1].metric("Training examples", training_report.get("supported_examples", "sample"))
    status_cols[2].metric("Unsupported", dataset_stats.get("unsupported_examples", 0))
    if evaluation_path.exists():
        status_cols[3].metric("Top-5 accuracy", f"{evaluation_report.get('top_5_template_accuracy', 0):.3f}")
    else:
        status_cols[3].metric("Top-5 accuracy", "Not measured")
        st.warning("Accuracy not yet measured — run evaluation after training.")
    status_cols[4].metric("Option A version", option_a_version)
    status_cols[5].metric("Hybrid router", "calibrated" if (_option_a_artifact_dir() / "hybrid_calibration.json").exists() else ("available" if option_a_ready else "disabled"))
    st.caption(f"Hybrid routing: {'enabled' if option_a_ready else 'disabled until Option A is trained'}")
    st.caption(f"Option A artifact path: {_option_a_artifact_dir()}")
    st.caption(f"Artifact path: {ARTIFACT_DIR}")
    summary_rows = [
        {"item": "datasets used", "value": ", ".join(training_report.get("datasets_used", [])) or "sample"},
        {"item": "train examples", "value": training_report.get("train_examples", "sample")},
        {"item": "validation examples", "value": training_report.get("validation_examples", "sample")},
        {"item": "test examples", "value": training_report.get("test_examples", "sample")},
        {"item": "unsupported examples", "value": dataset_stats.get("unsupported_examples", 0)},
        {"item": "top-1 accuracy", "value": f"{evaluation_report['top_1_template_accuracy']:.3f}" if "top_1_template_accuracy" in evaluation_report else "not measured"},
        {"item": "top-5 accuracy", "value": f"{evaluation_report['top_5_template_accuracy']:.3f}" if "top_5_template_accuracy" in evaluation_report else "not measured"},
        {"item": "training date", "value": training_report.get("training_date", "not trained")},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    by_template = training_report.get("by_template") or {}
    st.dataframe(
        pd.DataFrame([{"template": key, "examples": value} for key, value in sorted(by_template.items())]),
        use_container_width=True,
        hide_index=True,
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

with st.expander("Testing", expanded=False):
    if st.button("Run Golden Tests"):
        with st.spinner("Running golden tests..."):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_golden_tests.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            st.error("Golden tests failed to run.")
            st.code(completed.stderr or completed.stdout)
        else:
            st.success("Golden tests complete.")
            if completed.stdout:
                st.code(completed.stdout)

    golden = _golden_results()
    if golden:
        st.metric("Golden accuracy", f"{golden.get('accuracy', 0):.1%}")
        st.dataframe(pd.DataFrame(golden.get("case_results", [])), use_container_width=True, hide_index=True)
    else:
        st.caption("No golden test run yet.")

db_path_text = st.text_input("SQLite database path", value=str(DEFAULT_DB))
db_path = Path(db_path_text).expanduser()

left, right = st.columns([0.55, 0.45], vertical_alignment="top")

with left:
    question = st.text_input("Ask a question", value="Top 5 customers by sales")
    use_option_a_fallback = st.checkbox(
        "Use Option A fallback when Option C confidence is low",
        value=_option_a_ready(),
        disabled=not _option_a_ready(),
    )
    generate = st.button("Generate SQL", type="primary")

with right:
    st.caption("Local only: TF-IDF retrieval, QueryIR rendering, RapidFuzz matching, SQLGlot validation.")

if not db_path.exists():
    st.warning(f"Database not found: {db_path}")
    st.stop()

schema = read_sqlite_schema(db_path)
with st.expander("Schema", expanded=True):
    for table in schema.tables.values():
        cols = ", ".join(f"{col.name} ({col.type})" for col in table.columns.values())
        st.markdown(f"**{table.name}**: {cols}")

if generate and question.strip():
    try:
        model = _load_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error("Model could not be loaded. Run: python scripts/create_sample_db.py to create the database, then click Train From Local Datasets.")
        st.caption(str(exc))
        st.stop()
    if model.artifact_dir is None:
        st.info("Using sample model trained on hand-written examples. Train from datasets for better accuracy.")
    try:
        result = model.predict(question, schema, use_option_a_fallback=use_option_a_fallback)
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

    router_decision = result.router_decision or result.debug.get("router_decision") or {}
    option_a_result = result.option_a_result or result.debug.get("option_a_result") or {}

    st.subheader("Confidence")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Confidence", f"{result.confidence:.3f}")
    c2.metric("Tier", result.confidence_tier)
    c3.metric("Intent", result.intent or "unknown")
    c4.metric("Source model used", {"option_c": "Option C", "option_a": "Option A", "hybrid": "Hybrid"}.get(result.source_model, result.source_model))
    c5.metric("Option C confidence", f"{float(router_decision.get('option_c_confidence', result.confidence) or 0):.3f}")
    c6.metric("Option A calibrated", f"{float(router_decision.get('option_a_confidence', option_a_result.get('calibrated_confidence', 0)) or 0):.3f}")
    c7.metric("Option A version", result.option_a_version or option_a_result.get("option_a_version") or _option_a_version())
    if router_decision:
        st.caption(f"Router decision: {router_decision.get('selected')} ({router_decision.get('reason')})")
    repairs_applied = option_a_result.get("repairs_applied") or []
    if repairs_applied:
        st.info("Repairs applied: " + ", ".join(str(item) for item in repairs_applied))

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

    st.subheader("IR Validation")
    ir_validation = result.ir_validation or {}
    st.metric("IR validation", "passed" if ir_validation.get("is_valid") else "failed")
    if ir_validation.get("issues"):
        st.dataframe(pd.DataFrame(ir_validation.get("issues", [])), use_container_width=True, hide_index=True)
    else:
        st.caption("QueryIR created and IR validation passed.")

    st.subheader("Generated SQL")
    st.code(result.sql or "", language="sql")

    c1, c2, c3 = st.columns(3)
    c1.metric("Template", result.template_id or "unknown")
    c2.metric("Validation", "passed" if result.validation.get("is_valid", result.validation.get("ok")) else "failed")
    c3.metric("Selected example", (result.selected_candidate or {}).get("example_id", "none"))

    st.subheader("SQL Validation")
    checks = result.validation.get("checks") or {}
    st.dataframe(
        pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]),
        use_container_width=True,
        hide_index=True,
    )

    if result.validation.get("is_valid", result.validation.get("ok")):
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

    sql_is_valid = result.validation.get("is_valid", result.validation.get("ok", False))
    run_query = st.button("Run query", disabled=not sql_is_valid)
    if run_query and sql_is_valid and result.sql:
        try:
            df = execute_select(db_path, result.sql, validation_result=result.validation)
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

    if st.checkbox("Show QueryIR debug"):
        st.subheader("QueryIR JSON")
        st.json(result.query_ir or {})
        st.subheader("IR validation JSON")
        st.json(result.ir_validation or {})
        st.subheader("Router decision")
        st.json(router_decision or {"source_model": result.source_model, "option_a_tried": "option_a_result" in result.debug})
        st.subheader("Confidence comparison")
        st.json(
            {
                "option_c_confidence": router_decision.get("option_c_confidence"),
                "option_a_confidence": router_decision.get("option_a_confidence"),
                "selected": router_decision.get("selected"),
            }
        )
        st.subheader("Validation comparison")
        st.json(
            {
                "option_c_valid": router_decision.get("option_c_valid"),
                "option_a_valid": router_decision.get("option_a_valid"),
            }
        )
        if result.debug.get("option_c_result"):
            st.subheader("Option C result")
            st.json(result.debug.get("option_c_result"))
        if result.debug.get("option_a_result"):
            st.subheader("Option A result")
            st.json(result.debug.get("option_a_result"))
            option_a_debug = result.debug.get("option_a_result", {}).get("debug", {})
            st.subheader("Schema-link scores")
            st.json(option_a_debug.get("schema_linking", {}))
            st.subheader("Candidate pointer scores")
            st.json(option_a_debug.get("candidate_scores", {}))
            st.subheader("Raw decoded labels")
            st.json(option_a_debug.get("decoded_prediction", {}))
            st.subheader("Repaired QueryIR")
            st.json(result.debug.get("option_a_result", {}).get("repaired_query_ir", {}))
            st.subheader("Confidence calibration")
            st.json(option_a_debug.get("calibration", {}))
        st.subheader("PredictionResult JSON")
        st.json(result.model_dump())
elif not generate:
    st.info("Connect a SQLite database, ask a question, then generate SQL.")

if st.checkbox("Show sample table preview"):
    table_name = st.selectbox("Table", sorted(schema.tables))
    if table_name not in schema.tables:
        st.error("Selected table is not present in the connected schema.")
    else:
        preview_sql = build_safe_preview_sql(table_name, schema)
        if preview_sql is None:
            st.info("No safe preview columns available for this table.")
        else:
            df = execute_select(db_path, preview_sql)
            st.dataframe(pd.DataFrame(df), use_container_width=True)
