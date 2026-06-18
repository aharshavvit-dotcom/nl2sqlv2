from __future__ import annotations

from connected_db_testing.generated_case_runner import ConnectedDBRegressionRunner
from tests.test_60_schema_profiler import generic_schema


class BadJoinModel:
    def predict(self, question: str, schema: dict, case: dict | None = None) -> dict:
        return {"sql": 'SELECT * FROM "users" JOIN "assignments" ON "assignments"."user_id" = "users"."id" LIMIT 100'}


def test_connected_db_regression_runner_catches_unnecessary_join_select_star_and_sensitive_leak() -> None:
    case = {
        "case_id": "list_users",
        "question": "list all users",
        "case_type": "direct_table_listing",
        "expected": {
            "base_table": "users",
            "must_include": ['FROM "users"'],
            "must_not_include": ["JOIN", "password_hash"],
            "must_have_limit": True,
            "must_not_select_star": True,
        },
    }

    report = ConnectedDBRegressionRunner().run([case], generic_schema(), BadJoinModel())

    assert report["summary"]["case_pass_rate"] == 0.0
    assert report["summary"]["unnecessary_join_count"] >= 1
    assert report["summary"]["select_star_count"] == 1


def test_connected_db_regression_runner_passes_generated_smoke_cases() -> None:
    from connected_db_testing.schema_case_generator import SchemaCaseGenerator

    cases = SchemaCaseGenerator().generate_cases(generic_schema(), max_tables=3)
    report = ConnectedDBRegressionRunner().run(cases, generic_schema())

    assert report["summary"]["case_pass_rate"] == 1.0
