from __future__ import annotations

from execution_eval.sql_canonicalizer import SQLCanonicalizer


def test_canonicalizer_extracts_components_and_handles_quotes() -> None:
    result = SQLCanonicalizer().canonicalize('SELECT "users"."id" FROM "users" WHERE "users"."role" = \'admin\' LIMIT 10')

    assert "users" in result["tables"]
    assert any("id" in column for column in result["columns"])
    assert result["filters"]
    assert result["limit"] == 10


def test_canonicalizer_extracts_joins_and_handles_malformed_sql() -> None:
    joined = SQLCanonicalizer().canonicalize("SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100")
    malformed = SQLCanonicalizer().canonicalize("SELECT FROM")

    assert joined["joins"]
    assert malformed["parse_warnings"]
    assert "canonical_sql" in malformed
