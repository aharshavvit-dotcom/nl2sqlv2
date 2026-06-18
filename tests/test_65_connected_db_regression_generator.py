from __future__ import annotations

from connected_db_testing.schema_case_generator import SchemaCaseGenerator
from tests.test_60_schema_profiler import generic_schema


def test_connected_db_regression_generator_creates_direct_and_join_cases() -> None:
    cases = SchemaCaseGenerator().generate_cases(generic_schema())
    case_ids = {case["case_id"] for case in cases}

    assert "list_users" in case_ids
    assert "count_users" in case_ids
    assert any(case["case_type"] == "explicit_join" for case in cases)
    list_users = next(case for case in cases if case["case_id"] == "list_users")
    assert "JOIN" in list_users["expected"]["must_not_include"]
    assert "password_hash" in list_users["expected"]["must_not_include"]
