"""Generic join-policy enforcement."""

from __future__ import annotations

from generic_planner import JoinPolicy, infer_join_policy
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from tests.test_10_generic_table_intent import GENERIC_POSTGRES_SCHEMA


def test_join_policy_inference_for_generic_questions() -> None:
    assert infer_join_policy("list all users", "show_records") == JoinPolicy.NONE
    assert infer_join_policy("list all berth_masters", "show_records") == JoinPolicy.NONE
    assert infer_join_policy("show assignments with user names", "show_records") == JoinPolicy.EXPLICIT_ONLY
    assert infer_join_policy("assignments by user", "metric_by_dimension") == JoinPolicy.EXPLICIT_ONLY


def test_join_planner_returns_no_joins_for_none_policy() -> None:
    context = RuntimeSchemaContext(GENERIC_POSTGRES_SCHEMA)
    plan = RuntimeJoinPlanner().plan_joins(
        context,
        base_table="users",
        required_tables=["users", "assignments"],
        join_policy=JoinPolicy.NONE,
    )

    assert plan.join_policy == JoinPolicy.NONE.value
    assert plan.required_tables == ["users"]
    assert plan.join_steps == []
    assert plan.join_clause == ""
