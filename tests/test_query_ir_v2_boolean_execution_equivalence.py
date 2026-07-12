from __future__ import annotations

from ir.query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer
from ir.sql_to_query_ir_v2 import SQLToQueryIRV2Converter
from tests.query_ir_v2_boolean_helpers import execute_rows, load_boolean_eval_cases, sample_connection


def test_curated_boolean_eval_cases_parse_render_and_execute_equivalently() -> None:
    converter = SQLToQueryIRV2Converter()
    renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
    conn = sample_connection()

    for case in load_boolean_eval_cases():
        query = converter.convert(case["sql"])
        rendered = renderer.render(query)
        reparsed = converter.convert(rendered)

        assert reparsed.where.model_dump() == query.where.model_dump(), case["id"]  # type: ignore[union-attr]
        assert execute_rows(conn, rendered) == execute_rows(conn, case["sql"]), case["id"]
