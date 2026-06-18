from __future__ import annotations

from quality_gates.regression_suite import RegressionSuite


def test_regression_suite_catches_list_all_users_join_bug() -> None:
    def bad_predictor(question: str, schema: dict) -> dict:
        return {
            "query_ir": {
                "intent": "show_records",
                "base_table": "users",
                "required_tables": ["users", "assignments"],
                "joins": [{"condition": "assignments.user_id = users.id"}],
            },
            "sql": "SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100",
        }

    report = RegressionSuite(predictor=bad_predictor).run(
        cases=[
            {
                "case_id": "list_users",
                "question": "list all users",
                "schema_id": "generic_pg",
                "expected_intent": "show_records",
                "expected_base_table": "users",
                "expect_no_join": True,
            }
        ]
    )

    assert report["passed"] is False
    assert "unexpected join" in report["blocking_failures"][0]["issues"]
