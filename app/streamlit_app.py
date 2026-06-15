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

from db.connection_config import DatabaseConnectionConfig, safe_config_summary
from db.schema_reader import read_database_schema, schema_dict_to_graph, schema_summary
from execution.query_executor import execute_select, execute_query
from nl2sql_v1.feedback import append_feedback
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.verify_datasets import verify_all
from training.train_retriever_from_datasets import train_from_datasets


DEFAULT_DB = ROOT / "data" / "sample_retail.db"
EXAMPLES_PATH = ROOT / "training_data" / "examples.jsonl"
TEMPLATES_PATH = ROOT / "data" / "templates.yaml"
SYNONYMS_PATH = ROOT / "data" / "synonyms.yaml"
MODEL_PATH = ROOT / "models" / "tfidf_retriever.joblib"

# Resolve artifact dirs with fallback to old names
def _resolve_artifact_dir(new_name: str, old_name: str) -> Path:
    new_path = ROOT / "artifacts" / new_name
    return new_path if new_path.exists() else ROOT / "artifacts" / old_name

ARTIFACT_DIR = _resolve_artifact_dir("retrieval_ir_model", "option_c_model")
NEURAL_IR_ARTIFACT_DIR = _resolve_artifact_dir("neural_ir_model", "option_a_ir_model")
NEURAL_IR_V2_ARTIFACT_DIR = _resolve_artifact_dir("neural_ir_model", "option_a_ir_model_v2")
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


def _neural_ir_artifact_dir() -> Path:
    if _neural_ir_ready(NEURAL_IR_V2_ARTIFACT_DIR):
        return NEURAL_IR_V2_ARTIFACT_DIR
    if _neural_ir_ready(NEURAL_IR_ARTIFACT_DIR):
        return NEURAL_IR_ARTIFACT_DIR
    return NEURAL_IR_V2_ARTIFACT_DIR


def _neural_ir_ready(path: Path | None = None) -> bool:
    artifact_dir = path or _neural_ir_artifact_dir()
    return (
        (artifact_dir / "model.pt").exists()
        and (artifact_dir / "vocab.json").exists()
        and (artifact_dir / "label_maps.json").exists()
        and (artifact_dir / "config.yaml").exists()
    )


def _load_model() -> RetrievalNL2SQLModel:
    return RetrievalNL2SQLModel.load(
        artifact_dir=ARTIFACT_DIR,
        sample_model_path=MODEL_PATH,
        sample_examples_path=EXAMPLES_PATH,
        templates_path=TEMPLATES_PATH,
        synonyms_path=SYNONYMS_PATH,
        neural_ir_model_dir=_neural_ir_artifact_dir() if _neural_ir_ready() else None,
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


# ───────────────────────── Page Config ─────────────────────────
st.set_page_config(page_title="QueryIR NL-to-SQL", layout="wide")
st.title("QueryIR NL-to-SQL")

# ───────────────────────── Database Connection ─────────────────────────
with st.expander("Database Connection", expanded=True):
    db_type = st.radio("Database Type", ["SQLite", "PostgreSQL"], horizontal=True)

    if db_type == "SQLite":
        db_path_text = st.text_input("SQLite database path", value=str(DEFAULT_DB))
        db_config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=db_path_text)
        connect_clicked = st.button("Connect", key="connect_sqlite")
    else:
        col_h, col_p = st.columns(2)
        pg_host = col_h.text_input("Host", value="localhost")
        pg_port = col_p.number_input("Port", value=5432, min_value=1, max_value=65535)
        col_d, col_u = st.columns(2)
        pg_database = col_d.text_input("Database")
        pg_username = col_u.text_input("Username")
        pg_password = st.text_input("Password", type="password")
        col_ssl, col_schema = st.columns(2)
        pg_sslmode = col_ssl.selectbox("SSL Mode", ["prefer", "disable", "require"])
        pg_schema = col_schema.text_input("Schema", value="public")
        db_config = DatabaseConnectionConfig(
            db_type="postgres",
            host=pg_host,
            port=int(pg_port),
            database=pg_database,
            username=pg_username,
            password=pg_password,
            sslmode=pg_sslmode,
            schema_name=pg_schema,
        )
        connect_clicked = st.button("Connect", key="connect_pg")

    if connect_clicked:
        if db_config.db_type == "sqlite":
            path = Path(db_config.sqlite_path or "").expanduser()
            if not path.exists():
                st.error(f"Database not found: {path}")
                st.stop()
        # Test connection
        from db.sqlite_connector import SQLiteConnector
        if db_config.db_type == "sqlite":
            connector = SQLiteConnector(db_config)
        else:
            from db.postgres_connector import PostgresConnector
            connector = PostgresConnector(db_config)

        success, message = connector.test_connection()
        if success:
            st.session_state["db_config"] = db_config
            st.session_state["db_connected"] = True
            st.success(message)
        else:
            st.error(message)
            st.stop()
    elif "db_config" not in st.session_state:
        # Auto-connect to default SQLite if it exists
        if DEFAULT_DB.exists():
            st.session_state["db_config"] = DatabaseConnectionConfig(db_type="sqlite", sqlite_path=str(DEFAULT_DB))
            st.session_state["db_connected"] = True
        else:
            st.info("Connect a database to begin.")
            st.stop()

    # Show safe connection summary
    if "db_config" in st.session_state:
        summary = safe_config_summary(st.session_state["db_config"])
        st.caption(f"Connected: {summary.get('db_type')} — {summary.get('sqlite_path') or summary.get('database', '')}")

# ───────────────────────── Schema ─────────────────────────
active_config: DatabaseConnectionConfig = st.session_state.get("db_config")
if active_config is None:
    st.stop()

try:
    schema_dict = read_database_schema(active_config)
    schema = schema_dict_to_graph(schema_dict)
except Exception as exc:
    st.error(f"Failed to read schema: {exc}")
    st.stop()

with st.expander("Schema Summary", expanded=True):
    summary = schema_summary(schema_dict)
    cols = st.columns(5)
    cols[0].metric("Dialect", summary["dialect"])
    cols[1].metric("Database", summary.get("database") or "—")
    cols[2].metric("Tables", summary["table_count"])
    cols[3].metric("Columns", summary["column_count"])
    cols[4].metric("Relationships", summary["relationship_count"])

    if st.checkbox("Show table details"):
        for tbl_name, tbl_info in summary["tables"].items():
            pk_text = ", ".join(tbl_info["primary_keys"]) if tbl_info["primary_keys"] else "—"
            st.markdown(f"**{tbl_name}**: {tbl_info['column_count']} columns, PK: {pk_text}, FK: {tbl_info['foreign_key_count']}")

# ───────────────────────── Model Status ─────────────────────────
with st.expander("Model Status", expanded=False):
    evaluation_path = ARTIFACT_DIR / "evaluation_report.json"
    training_report = _load_json(ARTIFACT_DIR / "training_report.json")
    evaluation_report = _load_json(evaluation_path)
    dataset_stats = _load_json(ARTIFACT_DIR / "dataset_stats.json")
    retrieval_ir_ready = _artifact_ready()
    neural_ir_ready = _neural_ir_ready()
    status_cols = st.columns(4)
    status_cols[0].metric("Retrieval QueryIR Model", "trained" if retrieval_ir_ready else "sample")
    status_cols[1].metric("Training examples", training_report.get("supported_examples", "sample"))
    if evaluation_path.exists():
        status_cols[2].metric("Top-5 accuracy", f"{evaluation_report.get('top_5_template_accuracy', 0):.3f}")
    else:
        status_cols[2].metric("Top-5 accuracy", "Not measured")
    router_status = "calibrated" if (_neural_ir_artifact_dir() / "hybrid_calibration.json").exists() else ("available" if neural_ir_ready else "disabled")
    status_cols[3].metric("Adaptive QueryIR Router", router_status)
    st.caption(f"Neural QueryIR Model: {'ready' if neural_ir_ready else 'not trained'}")

# ───────────────────────── Dataset Training ─────────────────────────
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

# ───────────────────────── Testing ─────────────────────────
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
    else:
        st.caption("No golden test run yet.")

# ───────────────────────── Query ─────────────────────────
left, right = st.columns([0.55, 0.45], vertical_alignment="top")

with left:
    question = st.text_input("Ask a question", value="Top 5 customers by sales")
    use_neural_fallback = st.checkbox(
        "Use Neural fallback when retrieval confidence is low",
        value=_neural_ir_ready(),
        disabled=not _neural_ir_ready(),
    )
    generate = st.button("Generate SQL", type="primary")

with right:
    st.caption("Local only: TF-IDF retrieval, QueryIR rendering, RapidFuzz matching, SQLGlot validation.")

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
        result = model.predict(question, schema, use_neural_ir_fallback=use_neural_fallback)
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
    neural_ir_raw = result.neural_ir_result or result.debug.get("neural_ir_result") or {}

    st.subheader("Confidence")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    source_label = {
        "retrieval_ir": "Retrieval QueryIR", "neural_ir": "Neural QueryIR",
        "adaptive_router": "Adaptive Router",
        "option_c": "Retrieval QueryIR", "option_a": "Neural QueryIR", "hybrid": "Adaptive Router",
    }
    c1.metric("Confidence", f"{result.confidence:.3f}")
    c2.metric("Tier", result.confidence_tier)
    c3.metric("Intent", result.intent or "unknown")
    c4.metric("Source model", source_label.get(result.source_model, result.source_model))
    c5.metric("Retrieval QueryIR Confidence",
              f"{float(router_decision.get('retrieval_ir_confidence', router_decision.get('option_c_confidence', result.confidence)) or 0):.3f}")
    c6.metric("Neural QueryIR Confidence",
              f"{float(router_decision.get('neural_ir_confidence', router_decision.get('option_a_confidence', neural_ir_raw.get('calibrated_confidence', 0))) or 0):.3f}")
    if router_decision:
        st.caption(f"Router decision: {router_decision.get('selected')} ({router_decision.get('reason')})")
    repairs_applied = neural_ir_raw.get("repairs_applied") or []
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
    run_query = st.button("Run validated query", disabled=not sql_is_valid)
    if run_query and sql_is_valid and result.sql:
        try:
            df = execute_query(active_config, result.sql, validation_result=result.validation)
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
                "db_type": active_config.db_type,
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
        st.json(router_decision or {"source_model": result.source_model})
        st.subheader("Confidence comparison")
        st.json(
            {
                "retrieval_ir_confidence": router_decision.get("retrieval_ir_confidence", router_decision.get("option_c_confidence")),
                "neural_ir_confidence": router_decision.get("neural_ir_confidence", router_decision.get("option_a_confidence")),
                "selected": router_decision.get("selected"),
            }
        )
        if result.debug.get("retrieval_ir_result") or result.debug.get("option_c_result"):
            st.subheader("Retrieval QueryIR Result")
            st.json(result.debug.get("retrieval_ir_result") or result.debug.get("option_c_result"))
        if result.debug.get("neural_ir_result") or result.debug.get("option_a_result"):
            st.subheader("Neural QueryIR Result")
            neural_debug = result.debug.get("neural_ir_result") or result.debug.get("option_a_result", {})
            st.json(neural_debug)
            debug_inner = neural_debug.get("debug", {}) if isinstance(neural_debug, dict) else {}
            st.subheader("Schema-link scores")
            st.json(debug_inner.get("schema_linking", {}))
            st.subheader("Candidate pointer scores")
            st.json(debug_inner.get("candidate_scores", {}))
            st.subheader("Raw decoded labels")
            st.json(debug_inner.get("decoded_prediction", {}))
            st.subheader("Repaired QueryIR")
            st.json(neural_debug.get("repaired_query_ir", {}) if isinstance(neural_debug, dict) else {})
            st.subheader("Confidence calibration")
            st.json(debug_inner.get("calibration", {}))
        st.subheader("PredictionResult JSON")
        st.json(result.model_dump())
elif not generate:
    st.info("Connect a database, ask a question, then generate SQL.")
