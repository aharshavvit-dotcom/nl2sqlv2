from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_training.utils import read_jsonl, write_json
from execution_eval.execution_reporter import ExecutionReporter
from execution_eval.sql_structure_comparator import SQLStructureComparator
from inference.prediction_models import is_abstained_prediction
from ir.query_ir_models import diff_query_ir
from ir.sql_to_ir_converter import SQLToIRConverter
from validation.sql_validator import POLICY_FAILURE_TYPES, SQLValidator, policy_failure_type


def _schema(row: dict[str, Any]) -> dict[str, Any]:
    schema = row.get("schema") or row.get("schema_context") or {}
    if isinstance(schema, dict) and schema.get("tables"):
        return schema
    query_ir = row.get("gold_query_ir") or row.get("query_ir") or row.get("predicted_query_ir") or {}
    context = ((query_ir.get("metadata") or {}).get("validation_context") or {}).get("schema_context") or {}
    return {"dialect": query_ir.get("dialect") or row.get("dialect") or "sqlite", "tables": context.get("tables", {})}


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparator = SQLStructureComparator()
    evaluated = []
    for row in rows:
        predicted_sql = row.get("predicted_sql") or row.get("sql") or row.get("generated_sql") or ""
        gold_sql = row.get("gold_sql") or row.get("source_sql") or row.get("rendered_sql") or ""
        dialect = row.get("dialect") or _schema(row).get("dialect", "sqlite")
        structure = comparator.compare(predicted_sql, gold_sql, schema=_schema(row), dialect=dialect)
        evaluated.append(
            {
                "example_id": row.get("example_id"),
                "question": row.get("question"),
                "dataset_name": row.get("dataset_name"),
                "intent": (row.get("gold_query_ir") or row.get("query_ir") or {}).get("intent") or row.get("intent"),
                "execution_available": False,
                "execution_match": None,
                "structure": structure,
                "predicted_sql": predicted_sql,
                "gold_sql": gold_sql,
            }
        )
    return ExecutionReporter().summarize(evaluated)


def evaluate_controlled_fixtures(
    fixture_sql_path: Path | None = None,
    fixture_cases_path: Path | None = None,
) -> dict[str, Any]:
    """Run controlled execution-aware evaluation using a known fixture DB.

    Creates a temporary SQLite database from the fixture SQL seed,
    executes gold SQL for each case, and verifies row counts and safety.
    """
    fixture_dir = ROOT / "evaluation" / "fixtures"
    sql_path = fixture_sql_path or fixture_dir / "controlled_evaluation.sql"
    cases_path = fixture_cases_path or fixture_dir / "controlled_evaluation_cases.jsonl"

    if not sql_path.exists():
        raise FileNotFoundError(f"Fixture SQL not found: {sql_path}")
    if not cases_path.exists():
        raise FileNotFoundError(f"Fixture cases not found: {cases_path}")

    sql_seed = sql_path.read_text(encoding="utf-8")
    cases = read_jsonl(cases_path)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "controlled_evaluation.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(sql_seed)
            for case in cases:
                gold_sql = case.get("gold_sql", "")
                example_id = case.get("example_id", "")
                expected_rows = case.get("expected_row_count")
                entry: dict[str, Any] = {
                    "example_id": example_id,
                    "question": case.get("question"),
                    "gold_sql": gold_sql,
                    "expected_row_count": expected_rows,
                    "execution_success": False,
                    "actual_row_count": None,
                    "row_count_match": False,
                    "sql_is_select_only": gold_sql.strip().upper().startswith("SELECT"),
                    "error": None,
                }
                try:
                    cursor = conn.execute(gold_sql)
                    rows = cursor.fetchall()
                    entry["execution_success"] = True
                    entry["actual_row_count"] = len(rows)
                    if expected_rows is not None:
                        entry["row_count_match"] = len(rows) == expected_rows
                except Exception as exc:
                    entry["error"] = str(exc)
                results.append(entry)
        finally:
            conn.close()

    total = len(results)
    exec_ok = sum(1 for r in results if r["execution_success"])
    row_match = sum(1 for r in results if r["row_count_match"])
    select_only = sum(1 for r in results if r["sql_is_select_only"])

    return {
        "controlled_fixture_evaluation": True,
        "evaluation_type": "controlled_gold_sql_fixture_validation",
        "measures_model_predictions": False,
        "fixture_sql": str(sql_path),
        "fixture_cases": str(cases_path),
        "total_cases": total,
        "summary": {
            "execution_success_rate": exec_ok / total if total else 0.0,
            "row_count_match_rate": row_match / total if total else 0.0,
            "select_only_rate": select_only / total if total else 0.0,
        },
        "cases": results,
    }


def _build_schema_from_sqlite(
    conn: sqlite3.Connection,
) -> tuple[Any, list[Any]]:
    """Build a SchemaGraph by introspecting the live SQLite database.

    Returns (SchemaGraph, list_of_ForeignKeyInfo) so callers can inspect FK count.
    """
    from db.schema_graph import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo

    tables: dict[str, TableInfo] = {}
    all_fks: list[ForeignKeyInfo] = []

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_names = [row[0] for row in cursor.fetchall()]

    for table_name in table_names:
        columns: dict[str, ColumnInfo] = {}
        for col_row in conn.execute(f"PRAGMA table_info({table_name})").fetchall():
            # col_row: (cid, name, type, notnull, default_value, pk)
            col_name = col_row[1]
            col_type = col_row[2] or "TEXT"
            not_null = bool(col_row[3])
            is_pk = bool(col_row[5])
            columns[col_name] = ColumnInfo(
                name=col_name, type=col_type.lower(), nullable=not not_null, primary_key=is_pk,
            )

        fk_list: list[ForeignKeyInfo] = []
        for fk_row in conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall():
            # fk_row: (id, seq, table, from, to, on_update, on_delete, match)
            referred_table = fk_row[2]
            constrained_col = fk_row[3]
            referred_col = fk_row[4]
            fk = ForeignKeyInfo(
                table=table_name,
                constrained_column=constrained_col,
                referred_table=referred_table,
                referred_column=referred_col,
            )
            fk_list.append(fk)
            all_fks.append(fk)

        tables[table_name] = TableInfo(name=table_name, columns=columns, foreign_keys=fk_list)

    return SchemaGraph(tables=tables, dialect="sqlite"), all_fks


def _normalize_value(value: Any) -> Any:
    """Normalize a single value for comparison."""
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    """Normalize row values for order-independent comparison."""
    return sorted(
        tuple(_normalize_value(v) for v in row)
        for row in rows
    )


def _values_match(a: Any, b: Any, tolerance: float = 1e-6) -> bool:
    """Compare two values with float tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerance
    return a == b


def _rows_match_ordered(pred: list[tuple], gold: list[tuple], tolerance: float = 1e-6) -> bool:
    """Compare rows preserving order, with float tolerance."""
    if len(pred) != len(gold):
        return False
    for p_row, g_row in zip(pred, gold):
        if len(p_row) != len(g_row):
            return False
        if not all(_values_match(pv, gv, tolerance) for pv, gv in zip(p_row, g_row)):
            return False
    return True


def _compare_results(
    pred_rows: list[tuple],
    gold_rows: list[tuple],
    gold_sql: str,
) -> dict[str, Any]:
    """Normalized result comparison with order-awareness.

    If gold SQL has ORDER BY, uses ordered comparison.
    Otherwise, uses unordered normalized row-set comparison.
    """
    has_order_by = "ORDER BY" in gold_sql.upper()
    row_count_match = len(pred_rows) == len(gold_rows)

    # Unordered comparison (always computed)
    pred_normalized = _normalize_rows(pred_rows)
    gold_normalized = _normalize_rows(gold_rows)
    unordered_match = pred_normalized == gold_normalized

    # Ordered comparison (only meaningful with ORDER BY)
    ordered_match: bool | None = None
    if has_order_by:
        ordered_match = _rows_match_ordered(pred_rows, gold_rows)

    # Result value match: ordered when ORDER BY exists, unordered otherwise
    result_value_match = ordered_match if has_order_by else unordered_match

    return {
        "row_count_match": row_count_match,
        "unordered_result_match": unordered_match,
        "ordered_result_match": ordered_match,
        "result_value_match": result_value_match,
        "has_order_by": has_order_by,
    }


def _result_deltas(pred_rows: list[tuple], gold_rows: list[tuple], limit: int = 10) -> tuple[list[list[Any]], list[list[Any]]]:
    predicted = Counter(_normalize_rows(pred_rows))
    gold = Counter(_normalize_rows(gold_rows))
    missing = [list(row) for row, count in (gold - predicted).items() for _ in range(count)][:limit]
    extra = [list(row) for row, count in (predicted - gold).items() for _ in range(count)][:limit]
    return missing, extra


def _query_ir_from_sql(
    converter: SQLToIRConverter,
    question: str,
    sql: str,
    schema: Any,
    case_id: str,
) -> dict[str, Any]:
    if not sql:
        return {}
    converted = converter.convert(
        question=question,
        sql=sql,
        schema=schema,
        dataset_name="controlled-fixture",
        db_id="controlled",
        example_id=case_id,
        split="test",
    )
    return dict(converted.get("query_ir") or {}) if converted.get("success") else {}


def _projection_diagnostics(predicted_ir: dict[str, Any], gold_ir: dict[str, Any]) -> dict[str, Any]:
    def columns(payload: dict[str, Any]) -> set[str]:
        selected: set[str] = set()
        for section in ("dimensions", "metrics"):
            for item in payload.get(section) or []:
                if not isinstance(item, dict):
                    continue
                value = item.get("expression") or item.get("column")
                if value:
                    selected.add(str(value).strip().lower().replace('"', "").replace("`", ""))
        return selected

    predicted = columns(predicted_ir)
    gold = columns(gold_ir)
    extra = sorted(predicted - gold)
    return {
        "comparable": bool(predicted_ir and gold_ir),
        "predicted_columns": sorted(predicted),
        "gold_columns": sorted(gold),
        "exact_match": predicted == gold,
        "contains_gold": gold.issubset(predicted),
        "extra_columns": extra,
        "has_extra_columns": bool(extra),
        "default_projection_used": bool((predicted_ir.get("metadata") or {}).get("default_projection_used")),
    }


def _semantic_failure_category(
    query_ir_diff: dict[str, Any],
    predicted_sql: str,
    gold_sql: str,
    *,
    row_count_match: bool,
    result_value_match: bool,
    ordered_result_match: bool | None,
    dialect: str = "sqlite",
) -> str:
    slot = query_ir_diff.get("primary_failure_slot")
    slot_categories = {
        "filter_column": "filter_mismatch",
        "filter_value": "filter_mismatch",
        "filter_operator": "filter_mismatch",
        "projection": "projection_mismatch",
        "metric": "aggregation_mismatch",
        "aggregation": "aggregation_mismatch",
        "base_table": "join_mismatch",
        "join": "join_mismatch",
        "group_by": "group_by_mismatch",
        "order_by": "order_by_mismatch",
        "limit": "limit_mismatch",
        "date_filter": "filter_mismatch",
    }
    if slot in slot_categories:
        return slot_categories[slot]

    try:
        structure = SQLStructureComparator().compare(predicted_sql, gold_sql, dialect=dialect)
        scores = structure.get("component_scores") or {}
        for component, category in (
            ("selected_columns", "projection_mismatch"),
            ("filters", "filter_mismatch"),
            ("aggregations", "aggregation_mismatch"),
            ("group_by", "group_by_mismatch"),
            ("joins", "join_mismatch"),
            ("order_by", "order_by_mismatch"),
            ("limit", "limit_mismatch"),
        ):
            if float(scores.get(component, 1.0)) < 1.0:
                return category
    except Exception:
        pass
    if not row_count_match:
        return "row_count_mismatch"
    if ordered_result_match is False and "order by" in gold_sql.lower():
        return "order_by_mismatch"
    if not result_value_match:
        return "value_mismatch"
    return "unknown"


def evaluate_controlled_predicted_sql(
    model_artifact_dir: Path | None = None,
    fixture_sql_path: Path | None = None,
    fixture_cases_path: Path | None = None,
    config: dict[str, Any] | None = None,
    bundle_id: str | None = None,
    pipeline_run_id: str | None = None,
    candidate_bundle_dir: str | None = None,
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Run predicted-SQL controlled execution evaluation.

    Unlike evaluate_controlled_fixtures (which validates gold SQL),
    this function loads the trained model, generates predictions for each
    fixture question, and evaluates whether the predicted SQL executes
    correctly against the controlled fixture database.

    This is the stronger model metric: it measures actual model prediction accuracy.
    """
    from datetime import datetime, timezone

    fixture_dir = ROOT / "evaluation" / "fixtures"
    sql_path = fixture_sql_path or fixture_dir / "controlled_evaluation.sql"
    cases_path = fixture_cases_path or fixture_dir / "controlled_evaluation_cases.jsonl"

    # Identity metadata (Phase 1)
    identity = {
        "report_type": "controlled_predicted_sql_execution",
        "report_schema_version": "1.0",
        "bundle_id": bundle_id,
        "candidate_bundle_dir": candidate_bundle_dir,
        "model_artifact_source": "model_bundle_candidate" if candidate_bundle_dir else "artifact_dirs",
        "commit_sha": commit_sha or _git_commit_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": pipeline_run_id,
        "evaluation_mode": "real_model_predictions",
        "gold_replay_used": False,
        "is_valid_for_quality_gate": True,
    }

    if not sql_path.exists():
        return {**identity, "error": f"Fixture SQL not found: {sql_path}", "evaluation_type": "controlled_predicted_sql_execution"}
    if not cases_path.exists():
        return {**identity, "error": f"Fixture cases not found: {cases_path}", "evaluation_type": "controlled_predicted_sql_execution"}

    # Resolve model artifact directory
    artifact_dir = model_artifact_dir
    if artifact_dir is None:
        # Try candidate bundle, then current bundle, then raw artifacts
        for candidate_path in [
            ROOT / "artifacts" / "model_bundle" / "candidate",
            ROOT / "artifacts" / "model_bundle" / "current",
        ]:
            if (candidate_path / "retrieval_ir").exists():
                artifact_dir = candidate_path
                break
    if artifact_dir is None or not artifact_dir.exists():
        return {
            **identity,
            "error": "No model artifact directory found for predicted-SQL evaluation",
            "evaluation_type": "controlled_predicted_sql_execution",
            "measures_model_predictions": True,
        }
    identity["model_artifact_dir"] = str(artifact_dir)

    sql_seed = sql_path.read_text(encoding="utf-8")
    cases = read_jsonl(cases_path)

    # Load model
    try:
        from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
        model = RetrievalNL2SQLModel.load(str(artifact_dir))
    except Exception as exc:
        return {
            **identity,
            "error": f"Failed to load model from {artifact_dir}: {exc}",
            "evaluation_type": "controlled_predicted_sql_execution",
            "measures_model_predictions": True,
        }

    results: list[dict[str, Any]] = []
    schema_tables_available = 0
    schema_relationships_available = 0
    schema_graph_empty = True
    sql_validator = SQLValidator()
    ir_converter = SQLToIRConverter()
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "controlled_predicted.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(sql_seed)

            # Build real SchemaGraph from fixture database via introspection
            from nl2sql_v1.schema import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo
            schema, fk_list = _build_schema_from_sqlite(conn)
            schema_tables_available = len(schema.tables)
            schema_relationships_available = len(fk_list)
            schema_graph_empty = schema_tables_available == 0
            if schema_graph_empty:
                return {
                    **identity,
                    "error": "Schema graph is empty after SQLite introspection; cannot evaluate predicted SQL",
                    "evaluation_type": "controlled_predicted_sql_execution",
                    "measures_model_predictions": True,
                    "schema_graph_empty": True,
                    "schema_tables_available": 0,
                }

            # Execute gold SQL first to get expected results
            gold_results: dict[str, list[tuple]] = {}
            for case in cases:
                gold_sql = case.get("gold_sql", "")
                example_id = case.get("example_id", "")
                try:
                    cursor = conn.execute(gold_sql)
                    gold_results[example_id] = cursor.fetchall()
                except Exception:
                    gold_results[example_id] = []

            for case_index, case in enumerate(cases):
                example_id = case.get("example_id", "")
                question = case.get("question", "")
                gold_sql = case.get("gold_sql", "")
                expected_rows = case.get("expected_row_count")
                # Stable case_id (Phase 8)
                case_id = case.get("case_id") or f"controlled_{case_index + 1:03d}"
                gold_query_ir = _query_ir_from_sql(ir_converter, question, gold_sql, schema, f"{case_id}:gold")

                entry: dict[str, Any] = {
                    "case_id": case_id,
                    "example_id": example_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "expected_row_count": expected_rows,
                    "predicted_sql": None,
                    "prediction_generated": False,
                    "prediction_status": None,
                    "abstained": False,
                    "abstention_reason": None,
                    "requires_clarification": False,
                    "original_predicted_sql": None,
                    "sql_repair": {},
                    # Phase 3: Separated SQL validation from execution
                    "central_sql_validator_used": True,
                    "production_sql_valid": False,
                    "production_sql_validation_errors": [],
                    "production_sql_validation_warnings": [],
                    "blocked_by_production_policy": False,
                    "production_policy_blocks": [],
                    "policy_failure_type": None,
                    "failure_category": None,
                    "semantic_failure_category": None,
                    "fixture_execution_allowed": False,
                    "fixture_execution_blocked_reason": None,
                    "sqlite_execution_success": False,
                    "sqlite_execution_error": None,
                    "row_count_match": False,
                    "unordered_result_match": False,
                    "ordered_result_match": None,
                    "result_value_match": False,
                    "final_execution_match": False,
                    "gold_result_preview": [],
                    "predicted_result_preview": [],
                    "missing_rows_preview": [],
                    "extra_rows_preview": [],
                    "predicted_query_ir": {},
                    "gold_query_ir": gold_query_ir,
                    "query_ir_diff": {},
                    "projection_diagnostics": {},
                    # Legacy fields for backward compat
                    "predicted_sql_valid": False,
                    "predicted_sql_is_select_only": False,
                    "sql_validation_passed": False,
                    "sql_validation_errors": [],
                    "select_only": False,
                    "safe_sql": False,
                    "blocked_statement_reason": None,
                    "predicted_execution_success": False,
                    "predicted_actual_row_count": None,
                    "predicted_row_count_match": False,
                    "predicted_unordered_result_match": False,
                    "predicted_ordered_result_match": None,
                    "predicted_result_value_match": False,
                    "predicted_safe_sql": False,
                    "unsafe_sql": False,
                    "error": None,
                }

                # Generate prediction using the real fixture schema
                try:
                    result = model.predict(question, schema)
                    predicted_sql = result.sql or ""
                    result_status = getattr(result, "status", None)
                    abstention_reason = getattr(result, "abstention_reason", None)
                    if not isinstance(abstention_reason, str):
                        abstention_reason = None
                    result_debug = getattr(result, "debug", {})
                    if not isinstance(result_debug, dict):
                        result_debug = {}
                    entry["predicted_sql"] = predicted_sql
                    entry["prediction_generated"] = bool(predicted_sql.strip())
                    entry["prediction_status"] = result_status if isinstance(result_status, str) else None
                    entry["abstention_reason"] = abstention_reason
                    entry["requires_clarification"] = bool(
                        getattr(result, "needs_clarification", False) is True
                    )
                    entry["abstained"] = is_abstained_prediction(
                        sql=predicted_sql,
                        prediction_status=entry["prediction_status"],
                        requires_clarification=entry["requires_clarification"],
                    )
                    abstained = entry["abstained"]
                    predicted_query_ir = getattr(result, "query_ir", None) or getattr(result, "selected_query_ir", None)
                    if not isinstance(predicted_query_ir, dict):
                        predicted_query_ir = {}
                    if not predicted_query_ir and predicted_sql.strip():
                        predicted_query_ir = _query_ir_from_sql(
                            ir_converter, question, predicted_sql, schema, f"{case_id}:predicted"
                        )
                    entry["predicted_query_ir"] = predicted_query_ir
                    if predicted_query_ir and gold_query_ir:
                        entry["query_ir_diff"] = diff_query_ir(predicted_query_ir, gold_query_ir)
                        entry["projection_diagnostics"] = _projection_diagnostics(
                            predicted_query_ir, gold_query_ir,
                        )
                    entry["original_predicted_sql"] = (
                        result_debug.get("original_sql") or predicted_sql or None
                    )
                    entry["sql_repair"] = result_debug.get("sql_repair") or {}

                    if not predicted_sql.strip():
                        entry["failure_category"] = "abstained" if abstained else "no_prediction"
                        entry["fixture_execution_blocked_reason"] = (
                            abstention_reason or entry["failure_category"]
                        )
                        entry["error"] = (
                            f"Prediction abstained: {abstention_reason or 'unspecified_reason'}"
                            if abstained else "Model returned no SQL prediction"
                        )

                    if predicted_sql.strip():
                        validation = sql_validator.validate(
                            predicted_sql,
                            schema=schema,
                            dialect=getattr(schema, "dialect", "sqlite"),
                        )
                        select_only = bool((validation.get("checks") or {}).get("select_only", False))
                        validation_passed = bool(validation.get("is_valid", validation.get("ok", False)))
                        validation_errors = [str(item) for item in validation.get("issues", [])]
                        validation_warnings = [str(item) for item in validation.get("warnings", [])]

                        # Phase 10: Differentiate between execution/parse errors and production policy blocks
                        checks = validation.get("checks") or {}
                        parse_success = bool(checks.get("parse", False))
                        blocked_by_policy = parse_success and not validation_passed

                        # Phase 3: Production SQL validation result
                        entry["production_sql_valid"] = validation_passed
                        entry["production_sql_validation_errors"] = validation_errors
                        entry["production_sql_validation_warnings"] = validation_warnings
                        entry["blocked_by_production_policy"] = blocked_by_policy
                        entry["production_policy_blocks"] = [
                            k for k, v in checks.items() if not v and k != "parse"
                        ] if blocked_by_policy else []

                        # Legacy compat
                        entry["predicted_sql_is_select_only"] = select_only
                        entry["predicted_sql_valid"] = validation_passed
                        entry["sql_validation_passed"] = validation_passed
                        entry["sql_validation_errors"] = validation_errors
                        entry["select_only"] = select_only
                        entry["safe_sql"] = validation_passed
                        entry["predicted_safe_sql"] = validation_passed
                        entry["unsafe_sql"] = not validation_passed

                        if not validation_passed:
                            entry["policy_failure_type"] = policy_failure_type(validation)
                            entry["failure_category"] = "production_sql_validation_failed"
                            entry["blocked_statement_reason"] = _blocked_statement_reason(
                                validation, predicted_sql,
                            )
                            reason_prefix = "production_policy_blocked" if blocked_by_policy else "production_sql_validation_failed"
                            entry["fixture_execution_blocked_reason"] = (
                                f"{reason_prefix}: " + "; ".join(validation_errors)
                            )
                            entry["error"] = f"{reason_prefix}: " + "; ".join(validation_errors)
                        else:
                            # Phase 3: Fixture execution is allowed after validation
                            entry["fixture_execution_allowed"] = True
                            try:
                                cursor = conn.execute(predicted_sql)
                                pred_rows = cursor.fetchall()
                                entry["sqlite_execution_success"] = True
                                entry["predicted_execution_success"] = True
                                entry["predicted_actual_row_count"] = len(pred_rows)
                                if expected_rows is not None:
                                    entry["row_count_match"] = len(pred_rows) == expected_rows
                                    entry["predicted_row_count_match"] = entry["row_count_match"]
                                # Normalized result comparison
                                gold_rows = gold_results.get(example_id, [])
                                comparison = _compare_results(pred_rows, gold_rows, gold_sql)
                                entry["row_count_match"] = comparison["row_count_match"]
                                entry["predicted_row_count_match"] = comparison["row_count_match"]
                                entry["unordered_result_match"] = comparison["unordered_result_match"]
                                entry["ordered_result_match"] = comparison["ordered_result_match"]
                                entry["result_value_match"] = comparison["result_value_match"]
                                entry["final_execution_match"] = comparison["result_value_match"]
                                # Legacy compat
                                entry["predicted_unordered_result_match"] = comparison["unordered_result_match"]
                                entry["predicted_ordered_result_match"] = comparison["ordered_result_match"]
                                entry["predicted_result_value_match"] = comparison["result_value_match"]
                                entry["gold_result_preview"] = [list(row) for row in gold_rows[:10]]
                                entry["predicted_result_preview"] = [list(row) for row in pred_rows[:10]]
                                missing, extra = _result_deltas(pred_rows, gold_rows)
                                entry["missing_rows_preview"] = missing
                                entry["extra_rows_preview"] = extra
                                if not entry["final_execution_match"]:
                                    entry["semantic_failure_category"] = _semantic_failure_category(
                                        entry["query_ir_diff"],
                                        predicted_sql,
                                        gold_sql,
                                        row_count_match=entry["row_count_match"],
                                        result_value_match=entry["result_value_match"],
                                        ordered_result_match=entry["ordered_result_match"],
                                        dialect=getattr(schema, "dialect", "sqlite"),
                                    )
                                    entry["failure_category"] = entry["semantic_failure_category"]
                            except Exception as exc:
                                entry["sqlite_execution_error"] = str(exc)
                                entry["failure_category"] = "sqlite_execution_error"
                                entry["error"] = f"Execution error: {exc}"
                except Exception as exc:
                    entry["failure_category"] = "prediction_error"
                    entry["error"] = f"Prediction error: {exc}"

                entry["safety"] = {
                    "production_sql_valid": entry["production_sql_valid"],
                    "safe_sql": entry["safe_sql"],
                    "policy_failure_type": entry["policy_failure_type"],
                }
                entry["execution"] = {
                    "sqlite_execution_success": entry["sqlite_execution_success"],
                    "execution_error": entry["sqlite_execution_error"],
                }
                entry["semantic_match"] = {
                    "final_execution_match": entry["final_execution_match"],
                    "row_count_match": entry["row_count_match"],
                    "result_value_match": entry["result_value_match"],
                    "unordered_result_match": entry["unordered_result_match"],
                    "ordered_result_match": entry["ordered_result_match"],
                }

                results.append(entry)
        finally:
            conn.close()

    total = len(results)
    predictions_generated = sum(1 for r in results if r["prediction_generated"])
    production_valid = sum(1 for r in results if r["production_sql_valid"])
    production_invalid = sum(1 for r in results if r["prediction_generated"] and not r["production_sql_valid"])
    fixture_allowed = sum(1 for r in results if r["fixture_execution_allowed"])
    fixture_blocked = sum(1 for r in results if r["prediction_generated"] and not r["fixture_execution_allowed"])
    sqlite_success = sum(1 for r in results if r["sqlite_execution_success"])
    sqlite_error = sum(1 for r in results if r["fixture_execution_allowed"] and not r["sqlite_execution_success"])
    exec_match = sum(1 for r in results if r["final_execution_match"])
    row_match = sum(1 for r in results if r["row_count_match"])
    unordered_match = sum(1 for r in results if r.get("unordered_result_match"))
    ordered_match = sum(1 for r in results if r.get("ordered_result_match") is True)
    row_count_mismatch = sum(1 for r in results if r["sqlite_execution_success"] and not r["row_count_match"])
    value_mismatch = sum(1 for r in results if r["sqlite_execution_success"] and r["row_count_match"] and not r["result_value_match"])
    unsafe = sum(1 for r in results if r["prediction_generated"] and not r.get("safe_sql"))
    abstention_count = sum(1 for r in results if r.get("abstained"))
    requires_clarification_count = sum(1 for r in results if r.get("requires_clarification"))
    no_prediction_count = sum(
        1 for r in results
        if not r["prediction_generated"] and not r.get("abstained")
    )
    prediction_denominator = max(predictions_generated, 1)
    answered_count = sum(1 for r in results if r["prediction_generated"] and not r.get("abstained"))
    answered_denominator = max(answered_count, 1)
    safe_but_wrong_count = sum(
        1 for r in results
        if r.get("safe_sql") and r.get("sqlite_execution_success") and not r.get("final_execution_match")
    )
    semantic_categories = [
        "projection_mismatch",
        "filter_mismatch",
        "aggregation_mismatch",
        "group_by_mismatch",
        "join_mismatch",
        "order_by_mismatch",
        "limit_mismatch",
        "row_count_mismatch",
        "value_mismatch",
        "unknown",
    ]
    semantic_failure_breakdown = {
        category: sum(1 for row in results if row.get("semantic_failure_category") == category)
        for category in semantic_categories
    }
    projection_cases = [
        row.get("projection_diagnostics") or {}
        for row in results
        if (row.get("projection_diagnostics") or {}).get("comparable")
    ]
    projection_denominator = len(projection_cases)

    # Phase 3: Failure breakdown
    failure_breakdown = {
        "production_sql_validation_failed": production_invalid,
        "fixture_execution_blocked": fixture_blocked,
        "sqlite_execution_error": sqlite_error,
        "row_count_mismatch": row_count_mismatch,
        "value_mismatch": value_mismatch,
        "abstained": abstention_count,
        "no_prediction": no_prediction_count,
    }
    policy_failure_type_counts = {name: 0 for name in POLICY_FAILURE_TYPES}
    for result in results:
        failure_type = result.get("policy_failure_type")
        if failure_type in policy_failure_type_counts:
            policy_failure_type_counts[failure_type] += 1
    controlled_failure_diagnosis = [
        {
            "case_id": row.get("case_id"),
            "question": row.get("question"),
            "primary_failure": row.get("semantic_failure_category") or row.get("failure_category"),
            "primary_failure_slot": (row.get("query_ir_diff") or {}).get("primary_failure_slot"),
            "root_cause": _root_cause(row),
            "fix_area": _fix_area(row.get("semantic_failure_category") or row.get("failure_category")),
        }
        for row in results
        if not row.get("final_execution_match")
    ]

    return {
        # Phase 1: Identity metadata
        **identity,
        "evaluation_type": "controlled_predicted_sql_execution",
        "measures_model_predictions": True,
        "central_sql_validator_used": True,
        "fixture_sql": str(sql_path),
        "fixture_cases": str(cases_path),
        "schema_tables_available": schema_tables_available,
        "schema_relationships_available": schema_relationships_available,
        "schema_graph_empty": schema_graph_empty,
        "cases_total": total,
        "predictions_total": total,
        "predictions_generated": predictions_generated,
        "sql_generated_count": predictions_generated,
        "sql_evaluated_count": predictions_generated,
        "coverage_rate": answered_count / total if total else 0.0,
        "prediction_coverage_rate": answered_count / total if total else 0.0,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_count / total if total else 0.0,
        "requires_clarification_count": requires_clarification_count,
        "no_prediction_count": no_prediction_count,
        # Phase 3: Separated counts
        "production_sql_valid_count": production_valid,
        "production_sql_validation_failure_count": production_invalid,
        "fixture_execution_allowed_count": fixture_allowed,
        "fixture_execution_blocked_count": fixture_blocked,
        "sqlite_execution_success_count": sqlite_success,
        "sqlite_execution_error_count": sqlite_error,
        "predicted_execution_match_count": exec_match,
        "predicted_unordered_result_match_count": unordered_match,
        "predicted_ordered_result_match_count": ordered_match,
        "failure_breakdown": failure_breakdown,
        "semantic_failure_breakdown": semantic_failure_breakdown,
        "safe_but_wrong_sql_count": safe_but_wrong_count,
        "safe_but_wrong_sql_rate": safe_but_wrong_count / production_valid if production_valid else 0.0,
        "projection_exact_match_rate": (
            sum(bool(item.get("exact_match")) for item in projection_cases) / projection_denominator
            if projection_denominator else 0.0
        ),
        "projection_contains_gold_rate": (
            sum(bool(item.get("contains_gold")) for item in projection_cases) / projection_denominator
            if projection_denominator else 0.0
        ),
        "extra_projection_column_rate": (
            sum(bool(item.get("has_extra_columns")) for item in projection_cases) / projection_denominator
            if projection_denominator else 0.0
        ),
        "default_projection_used_count": sum(
            bool(item.get("default_projection_used")) for item in projection_cases
        ),
        "semantic_execution_match_rate": exec_match / answered_denominator if answered_count else 0.0,
        "quality_on_answered_rate": exec_match / answered_denominator if answered_count else 0.0,
        "quality_on_all_questions_rate": exec_match / total if total else 0.0,
        "controlled_failure_diagnosis": controlled_failure_diagnosis,
        "policy_failure_type_counts": policy_failure_type_counts,
        # Rates
        "predicted_sql_valid_count": production_valid,
        "predicted_execution_success_count": sqlite_success,
        "predicted_sql_validation_success_rate": production_valid / prediction_denominator,
        "predicted_safe_sql_rate": production_valid / prediction_denominator,
        "predicted_select_only_rate": sum(1 for r in results if r.get("select_only")) / prediction_denominator,
        "predicted_unsafe_sql_count": unsafe,
        "predicted_execution_success_rate": sqlite_success / answered_denominator if answered_count else 0.0,
        "predicted_execution_match_rate": exec_match / answered_denominator if answered_count else 0.0,
        "predicted_execution_error_rate": sqlite_error / fixture_allowed if fixture_allowed else 0.0,
        "predicted_row_count_match_rate": row_match / answered_denominator if answered_count else 0.0,
        "predicted_result_value_match_rate": exec_match / answered_denominator if answered_count else 0.0,
        "unsafe_sql_count": unsafe,
        "passed": bool(
            predictions_generated > 0
            and unsafe == 0
            and (exec_match / answered_denominator if answered_count else 0.0)
            >= (config or {}).get("min_execution_match_rate", 0.0)
        ),
        "cases": results,
    }


def _fix_area(category: str | None) -> str:
    return {
        "projection_mismatch": "default_projection",
        "filter_mismatch": "filter_grounding",
        "aggregation_mismatch": "metric_grounding",
        "group_by_mismatch": "dimension_grounding",
        "join_mismatch": "join_planning",
        "order_by_mismatch": "ordering",
        "limit_mismatch": "limit_resolution",
        "row_count_mismatch": "filter_or_join_grounding",
        "value_mismatch": "semantic_grounding",
    }.get(str(category), "prediction_pipeline")


def _root_cause(row: dict[str, Any]) -> str:
    slot = (row.get("query_ir_diff") or {}).get("primary_failure_slot")
    if slot:
        return f"QueryIR {slot} differs from gold"
    if row.get("semantic_failure_category") == "row_count_mismatch":
        return "predicted result row count differs from gold"
    if row.get("semantic_failure_category") == "value_mismatch":
        return "predicted rows have different values from gold"
    return str(row.get("error") or row.get("failure_category") or "unknown semantic mismatch")


def _git_commit_sha() -> str | None:
    """Try to get current git commit SHA."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip() or None
    except Exception:
        return None


def _blocked_statement_reason(validation: dict[str, Any], sql: str) -> str:
    checks = validation.get("checks") or {}
    if not checks.get("select_only", False):
        return "non_select_statement"
    if not checks.get("single_statement", True):
        return "multiple_statements"
    if not checks.get("no_blocked_keywords", True):
        return "blocked_keyword"
    if not checks.get("no_comments", True):
        return "comments_not_allowed"
    if not checks.get("limit_present", True) or not checks.get("limit_within_bounds", True):
        return "limit_policy_failed"
    if not checks.get("tables_exist", True) or not checks.get("columns_exist", True):
        return "schema_validation_failed"
    if not checks.get("no_sensitive_columns", True):
        return "sensitive_column"
    if not checks.get("no_dangerous_functions", True):
        return "dangerous_function"
    return "sql_validation_failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-aware evaluation on prediction rows.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "self_training" / "validation_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "execution_aware_evaluation_report.json")
    parser.add_argument("--run-controlled-fixtures", action="store_true", help="Run controlled fixture evaluation instead of prediction rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_controlled_fixtures:
        report = evaluate_controlled_fixtures()
        write_json(args.output, report)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
        return 0
    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}. Run training/evaluate_against_gold.py first.")
    rows = read_jsonl(args.predictions)
    report = evaluate_rows(rows)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
