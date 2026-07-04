from __future__ import annotations

import json
import hashlib
import os
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
from feedback.feedback_models import QueryFeedback
from feedback.feedback_store import FeedbackStore
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from connected_db_testing.generated_case_runner import ConnectedDBRegressionReporter, ConnectedDBRegressionRunner
from connected_db_testing.schema_case_generator import SchemaCaseGenerator, write_cases_jsonl
from semantic_layer import build_semantic_profile
from semantic_layer.semantic_profile_store import SemanticProfileStore
from scripts.verify_datasets import verify_all
from model_bundle.bundle_loader import ModelBundleLoader, inspect_bundle_status

# ──────────────────── Developer Config Flag ────────────────────
# Set to True to show training UI in the app (developer mode only).
# Normal users should train via: python training/train_model.py --config configs/training.yaml
ENABLE_DEV_TRAINING_UI = False


DEFAULT_DB = ROOT / "data" / "sample_retail.db"
EXAMPLES_PATH = ROOT / "training_data" / "examples.jsonl"
TEMPLATES_PATH = ROOT / "data" / "templates.yaml"
SYNONYMS_PATH = ROOT / "data" / "synonyms.yaml"
MODEL_PATH = ROOT / "models" / "tfidf_retriever.joblib"

# Default bundle path
DEFAULT_BUNDLE_DIR = ROOT / "artifacts" / "model_bundle" / "current"
DEFAULT_CANDIDATE_BUNDLE_DIR = ROOT / "artifacts" / "model_bundle" / "candidate"

ARTIFACT_DIR = ROOT / "artifacts" / "work" / "retrieval_ir"
NEURAL_IR_ARTIFACT_DIR = ROOT / "artifacts" / "work" / "neural_ir"
NEURAL_IR_V2_ARTIFACT_DIR = ROOT / "artifacts" / "work" / "neural_ir"
FEEDBACK_PATH = ROOT / "data" / "feedback" / "query_feedback.jsonl"
EVALUATION_DIR = ROOT / "evaluation"
GOLDEN_RESULTS_PATH = EVALUATION_DIR / "golden_runtime_report.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_fingerprint(schema_payload: dict[str, Any]) -> str:
    safe_payload = {
        "dialect": schema_payload.get("dialect"),
        "schema_name": schema_payload.get("schema_name"),
        "tables": {
            table: [
                {
                    "name": column.get("name"),
                    "type": column.get("type"),
                    "is_primary_key": column.get("is_primary_key"),
                }
                for column in info.get("columns", [])
            ]
            for table, info in sorted((schema_payload.get("tables") or {}).items())
        },
    }
    encoded = json.dumps(safe_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _try_load_bundle(
    bundle_dir: Path,
    *,
    allow_candidate_debug: bool = False,
) -> dict[str, Any] | None:
    """Try to load a model bundle. Returns bundle info or None."""
    st.session_state.pop("bundle_load_error", None)
    try:
        loader = ModelBundleLoader()
        return loader.load(bundle_dir, allow_candidate_debug=allow_candidate_debug)
    except (FileNotFoundError, ValueError) as exc:
        st.session_state["bundle_load_error"] = _format_bundle_load_error(exc)
        return None
    except Exception as exc:
        st.session_state["bundle_load_error"] = _format_bundle_load_error(exc)
        return None


def _format_bundle_load_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Invalid model bundle:"):
        raw = message.split(":", 1)[1]
        if "\n-" in raw:
            issues = [item.strip().lstrip("-").strip() for item in raw.splitlines() if item.strip().startswith("-")]
        else:
            issues = [item.strip() for item in raw.split(";") if item.strip()]
        if issues:
            return "Bundle invalid:\n" + "\n".join(f"- {issue}" for issue in issues)
    return f"Bundle invalid:\n- {message or exc.__class__.__name__}"


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


def _load_model_from_bundle(bundle: dict[str, Any]) -> RetrievalNL2SQLModel:
    """Load model using bundle-resolved paths."""
    retrieval_dir = Path(bundle["retrieval_model_dir"])
    neural_dir = Path(bundle["neural_model_dir"])
    neural_ready = (neural_dir / "model.pt").exists()
    if not retrieval_dir.exists():
        raise FileNotFoundError(f"Bundle retrieval artifact directory missing: {retrieval_dir}")

    return RetrievalNL2SQLModel.load(
        artifact_dir=retrieval_dir,
        sample_model_path=MODEL_PATH,
        sample_examples_path=EXAMPLES_PATH,
        templates_path=TEMPLATES_PATH,
        synonyms_path=SYNONYMS_PATH,
        neural_ir_model_dir=neural_dir if neural_ready else None,
        allow_dev_fallback=False,
    )


def _load_model_legacy() -> RetrievalNL2SQLModel:
    """Legacy model loading using artifact folder guessing (dev mode only)."""
    return RetrievalNL2SQLModel.load(
        artifact_dir=ARTIFACT_DIR,
        sample_model_path=MODEL_PATH,
        sample_examples_path=EXAMPLES_PATH,
        templates_path=TEMPLATES_PATH,
        synonyms_path=SYNONYMS_PATH,
        neural_ir_model_dir=_neural_ir_artifact_dir() if _neural_ir_ready() else None,
        allow_dev_fallback=True,
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

# ───────────────────────── Model Bundle ─────────────────────────
with st.sidebar:
    st.subheader("Model Bundle")
    bundle_status = inspect_bundle_status(DEFAULT_BUNDLE_DIR, DEFAULT_CANDIDATE_BUNDLE_DIR)
    st.caption(
        "Production bundle status: "
        f"current={'found' if bundle_status['current_bundle_found'] else 'not found'}, "
        f"candidate={'found' if bundle_status['candidate_bundle_found'] else 'not found'}"
    )
    if bundle_status["last_quality_gate_passed"] is False:
        st.warning("Last candidate quality gate failed.")
        for index, blocker in enumerate(bundle_status["top_blockers"], start=1):
            st.caption(f"{index}. {blocker}")
    bundle_path_input = st.text_input(
        "Model Bundle Path",
        value=str(DEFAULT_BUNDLE_DIR),
        help="Path to a validated model bundle directory",
    )
    bundle_dir = Path(bundle_path_input)
    env_candidate_debug = os.getenv("NL2SQL_ALLOW_CANDIDATE_BUNDLE", "0") == "1"
    allow_candidate_debug = st.checkbox(
        "Use candidate bundle for debugging",
        value=env_candidate_debug,
        help="Never promotes the candidate or marks it production-ready.",
    )
    bundle_info = _try_load_bundle(
        bundle_dir,
        allow_candidate_debug=allow_candidate_debug,
    )
    if bundle_info is None and allow_candidate_debug and bundle_dir == DEFAULT_BUNDLE_DIR:
        bundle_info = _try_load_bundle(
            DEFAULT_CANDIDATE_BUNDLE_DIR,
            allow_candidate_debug=True,
        )

    if bundle_info:
        manifest = bundle_info.get("manifest", {})
        st.success(f"Bundle loaded: {manifest.get('bundle_id', 'unknown')}")
        st.caption(f"Status: {manifest.get('status', 'unknown')}")
        if bundle_info.get("loaded_for_debug"):
            st.warning(
                "Warning: Candidate bundle loaded for debugging only. "
                "This model did not pass production quality gate."
            )
    else:
        st.warning(
            "No validated model bundle found.\n\n"
            "Run:\n```\npython training/train_model.py \\\n  --config configs/training.yaml\n```"
        )
        if st.session_state.get("bundle_load_error"):
            st.error(st.session_state["bundle_load_error"])
        if ENABLE_DEV_TRAINING_UI:
            st.info("Developer mode: falling back to legacy artifact loading.")

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
    semantic_profile = build_semantic_profile(schema_dict)
    SemanticProfileStore(ROOT / "artifacts" / "semantic_profiles").save(semantic_profile["schema_fingerprint"], semantic_profile)
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

    st.subheader("Semantic Profile")
    table_infos = semantic_profile.get("tables") or {}
    semantic_cols = st.columns(5)
    semantic_cols[0].metric("Entity tables", sum(1 for info in table_infos.values() if info.get("table_type") == "entity"))
    semantic_cols[1].metric("Master tables", sum(1 for info in table_infos.values() if info.get("table_type") == "lookup"))
    semantic_cols[2].metric("Assignment tables", sum(1 for info in table_infos.values() if info.get("table_type") == "bridge"))
    semantic_cols[3].metric("Metrics", len(semantic_profile.get("metrics") or {}))
    semantic_cols[4].metric("Sensitive columns", sum(len(info.get("sensitive_columns") or []) for info in table_infos.values()))
    with st.expander("Semantic profile details", expanded=False):
        st.json(
            {
                "entities": [table for table, info in table_infos.items() if info.get("table_type") == "entity"],
                "masters": [table for table, info in table_infos.items() if info.get("table_type") == "lookup"],
                "assignments": [table for table, info in table_infos.items() if info.get("table_type") == "bridge"],
                "metrics": semantic_profile.get("metrics") or {},
                "dimensions": semantic_profile.get("dimensions") or {},
                "dates": semantic_profile.get("dates") or {},
                "sensitive_columns": {
                    table: info.get("sensitive_columns") or []
                    for table, info in table_infos.items()
                    if info.get("sensitive_columns")
                },
            }
        )
    regression_dir = ROOT / "artifacts" / "connected_db_regressions"
    cases_path = regression_dir / "generated_cases.jsonl"
    if st.button("Generate connected-DB regression cases"):
        cases = SchemaCaseGenerator().generate_cases(schema_dict)
        write_cases_jsonl(str(cases_path), cases)
        st.success(f"Generated {len(cases)} cases.")
    if st.button("Run connected-DB regression smoke test"):
        if cases_path.exists():
            cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            cases = SchemaCaseGenerator().generate_cases(schema_dict)
            write_cases_jsonl(str(cases_path), cases)
        report = ConnectedDBRegressionRunner().run(cases, schema_dict)
        ConnectedDBRegressionReporter().write(report, regression_dir / "regression_report.json")
        st.json(report.get("summary", {}))

# ───────────────────────── Model Status ─────────────────────────
with st.expander("Model Status", expanded=False):
    if bundle_info:
        manifest = bundle_info.get("manifest", {})
        status_cols = st.columns(4)
        status_cols[0].metric("Bundle Status", manifest.get("status", "unknown"))
        status_cols[1].metric("Bundle ID", manifest.get("bundle_id", "unknown")[:20])
        qg = manifest.get("quality_gate", {})
        status_cols[2].metric("Quality Gate", "passed" if qg.get("passed") else "not passed")
        metrics = manifest.get("metrics", {})
        status_cols[3].metric("SQL Validation Rate", f"{metrics.get('sql_validation_rate', 0):.1%}")
    else:
        st.error("No validated model bundle loaded.")
        if st.session_state.get("bundle_load_error"):
            st.error(st.session_state["bundle_load_error"])
        st.code("python training/train_model.py --config configs/training.yaml", language="bash")

# ───────────────────────── Dataset Training (Developer Mode Only) ─────────────────────────
if ENABLE_DEV_TRAINING_UI:
    with st.expander("Dataset Training (Developer Mode)", expanded=False):
        st.warning("⚠️ Developer mode training. For production, use: `python training/train_model.py --config configs/training.yaml`")
        from training.train_retriever_from_datasets import train_from_datasets

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
        if bundle_info:
            model = _load_model_from_bundle(bundle_info)
        elif ENABLE_DEV_TRAINING_UI:
            model = _load_model_legacy()
        else:
            st.error(
                "No validated model bundle found. Run:\n\n"
                "```\npython training/train_model.py --config configs/training.yaml\n```"
            )
            st.stop()
    except (FileNotFoundError, ValueError) as exc:
        st.error(_format_bundle_load_error(exc))
        st.code("python training/train_model.py --config configs/training.yaml", language="bash")
        st.stop()
    try:
        result = model.predict(question, schema, use_neural_ir_fallback=use_neural_fallback)
    except Exception as exc:
        st.error(f"Could not generate SQL for this schema: {exc}")
        st.stop()

    if result.needs_clarification:
        st.warning("The question is ambiguous.")
        clarification = result.clarification or {}
        options = clarification.get("options") or []
        st.write("Please choose one option:")
        selected_option = st.radio("Clarification options", options, label_visibility="collapsed") if options else None
        if st.button("Continue", disabled=not bool(selected_option)):
            st.session_state["last_clarification_choice"] = {
                "question": question,
                "selected_option": selected_option,
                "clarification": clarification,
            }
            st.info(f"Clarification selected: {selected_option}")
        with st.expander("Clarification Debug", expanded=True):
            st.json(
                {
                    "ambiguity_type": clarification.get("ambiguity_type"),
                    "candidate_mappings": clarification.get("candidate_mappings"),
                    "scores": clarification.get("scores"),
                    "reason": clarification.get("reason"),
                }
            )
        st.stop()

    st.subheader("Retrieved Examples")
    if result.retrieved_candidates:
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
    else:
        st.caption("No retrieval examples used.")

    router_decision = result.router_decision or result.debug.get("router_decision") or {}
    neural_ir_raw = result.neural_ir_result or result.debug.get("neural_ir_result") or {}

    # Runtime metadata
    with st.expander("Runtime Metadata", expanded=False):
        runtime_cols = st.columns(4)
        runtime_source = result.debug.get("runtime_source", "unknown")
        dev_fallback = result.debug.get("dev_fallback_used", False)
        runtime_cols[0].metric("Runtime Source", runtime_source)
        runtime_cols[1].metric("Dev Fallback", "Yes" if dev_fallback else "No")
        runtime_cols[2].metric("Calibrated Conf.", f"{result.calibrated_confidence:.3f}" if result.calibrated_confidence is not None else "N/A")
        runtime_cols[3].metric("Abstain", "Yes" if result.abstain else "No")

        # Calibration and drift loading status
        extra_cols = st.columns(4)
        extra_cols[0].metric("Calibration Loaded", "Yes" if result.debug.get("calibration_loaded") else "No")
        extra_cols[1].metric("Drift Baseline Loaded", "Yes" if result.debug.get("schema_drift_baseline_loaded") else "No")
        extra_cols[2].metric("Raw Confidence", f"{result.raw_confidence:.3f}" if result.raw_confidence is not None else "N/A")
        extra_cols[3].metric("Conformal Threshold", f"{result.conformal_threshold:.3f}" if result.conformal_threshold is not None else "N/A")

        if bundle_info:
            cal_path = bundle_info.get("calibration_report_path")
            st.caption(f"Bundle ID: {result.debug.get('bundle_id') or bundle_info.get('bundle_id', 'N/A')}")
            st.caption(f"Bundle dir: {result.debug.get('bundle_dir') or bundle_info.get('bundle_dir', 'N/A')}")
            st.caption(f"Bundle status: {result.debug.get('bundle_status') or bundle_info.get('status', 'N/A')}")
            st.caption(f"Calibration report: {'loaded' if cal_path else 'not available'}")
        if dev_fallback:
            st.warning("⚠️ Development fallback is active. This is not production-safe. "
                        "Run `python training/train_model.py` to build a validated model bundle.")
        # Abstention reason
        if result.abstain and result.abstention_reason:
            st.info(f"Abstention reason: {result.abstention_reason}")
        elif result.abstain:
            st.info("Abstention triggered (no specific reason recorded)")
        # Schema drift flags
        if result.schema_drift_flags:
            st.warning("Schema drift detected:\n" + "\n".join(f"- {flag}" for flag in result.schema_drift_flags))

    st.subheader("Confidence")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    source_label = {
        "generic_direct_planner": "Generic Direct Planner",
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
    if result.source_model == "generic_direct_planner":
        planner_debug = result.planner_debug or result.debug.get("generic_planner") or {}
        st.info("Direct schema-safe query detected")
        st.caption(f"No joins required. Base table: {planner_debug.get('base_table') or (result.query_ir or {}).get('base_table')}")
        with st.expander("Generic Planner Debug", expanded=False):
            st.json(planner_debug)
    repairs_applied = neural_ir_raw.get("repairs_applied") or []
    if repairs_applied:
        st.info("Repairs applied: " + ", ".join(str(item) for item in repairs_applied))

    st.subheader("Slots")
    if result.slots:
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
    else:
        st.caption("Direct planner did not require slot extraction.")

    st.subheader("Schema Mapping")
    mapping = result.schema_mapping
    mapping_rows = [
        {"item": "base", "table": mapping.get("base_table"), "column": None, "score": (mapping.get("match_scores") or {}).get("generic_direct_planner")},
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

    with st.expander("Optional: Manual Feedback (Legacy)", expanded=False):
        st.info("The primary training loop now uses dataset-driven self-improvement. Manual feedback is optional.")
        st.subheader("Feedback")
        rating_label = st.radio(
            "Was this answer useful?",
            ["Correct", "Partially correct", "Incorrect", "Unsafe", "Not sure"],
            horizontal=True,
        )
        tag_options = {
            "wrong table": "wrong_table",
            "wrong join": "wrong_join",
            "unnecessary join": "unnecessary_join",
            "wrong metric": "wrong_metric",
            "wrong dimension": "wrong_dimension",
            "missing filter": "missing_filter",
            "wrong filter": "wrong_filter",
            "invalid SQL": "invalid_sql",
            "unsafe SQL": "unsafe_sql",
        }
        selected_tag_labels = st.multiselect("What was wrong?", list(tag_options))
        corrected_sql = st.text_area("Corrected SQL, optional", height=90)
        comment = st.text_area("Comment, optional", height=80)
        if st.button("Submit feedback"):
            rating_map = {
                "Correct": "correct",
                "Partially correct": "partially_correct",
                "Incorrect": "incorrect",
                "Unsafe": "unsafe",
                "Not sure": "not_sure",
            }
            feedback = QueryFeedback(
                db_type=active_config.db_type,
                schema_fingerprint=_schema_fingerprint(schema_dict),
                question=question,
                generated_query_ir=result.query_ir,
                generated_sql=result.sql,
                source_model=result.source_model,
                validation_status=result.validation,
                execution_status=None,
                user_rating=rating_map[rating_label],
                user_comment=comment or None,
                corrected_sql=corrected_sql.strip() or None,
                corrected_query_ir=None,
                feedback_tags=[tag_options[label] for label in selected_tag_labels],
            )
            feedback_id = FeedbackStore(FEEDBACK_PATH).append(feedback)
            st.success(f"Feedback saved: {feedback_id}")

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
