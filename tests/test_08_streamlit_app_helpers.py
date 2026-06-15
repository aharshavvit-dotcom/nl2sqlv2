"""Test 08: Streamlit App Helpers — connection config, schema summary, naming compliance."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from db.connection_config import DatabaseConnectionConfig, safe_config_summary
from db.schema_reader import schema_summary


ROOT = Path(__file__).resolve().parents[1]


class TestConnectionConfigForm:
    def test_sqlite_config_creation(self) -> None:
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path="/tmp/test.db")
        assert config.db_type == "sqlite"

    def test_postgres_config_creation(self) -> None:
        config = DatabaseConnectionConfig(db_type="postgres", host="localhost",
                                          port=5432, database="mydb",
                                          username="user", password="pass")
        assert config.db_type == "postgres"
        assert config.schema_name == "public"


class TestPasswordMasking:
    def test_safe_summary_masks_password(self) -> None:
        config = DatabaseConnectionConfig(db_type="postgres", host="h", port=5432,
                                          database="db", username="u", password="s3cr3t!")
        summary = safe_config_summary(config)
        assert summary["password"] == "***"
        assert "s3cr3t!" not in str(summary)

    def test_sqlalchemy_url_is_private(self) -> None:
        """Ensure we don't accidentally expose passwords in config summaries."""
        config = DatabaseConnectionConfig(db_type="postgres", host="h", port=5432,
                                          database="db", username="u", password="s3cr3t!")
        url = config.sqlalchemy_url()
        assert "s3cr3t!" in url  # It's in the URL but...
        summary = safe_config_summary(config)
        assert "s3cr3t!" not in str(summary)  # ...never in the summary


class TestSchemaSummary:
    def test_summary_shape(self) -> None:
        schema_dict = {
            "dialect": "sqlite",
            "database": "test.db",
            "schema_name": None,
            "tables": {
                "orders": {
                    "columns": [{"name": "id"}, {"name": "amount"}],
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                },
            },
            "relationships": [],
        }
        summary = schema_summary(schema_dict)
        assert summary["table_count"] == 1
        assert summary["column_count"] == 2
        assert "orders" in summary["tables"]


class TestUILabelNaming:
    """Verify no 'Option A' or 'Option C' labels leak into the Streamlit app."""

    def test_no_option_a_c_in_streamlit_app(self) -> None:
        app_path = ROOT / "app" / "streamlit_app.py"
        source = app_path.read_text(encoding="utf-8")
        # These patterns should not appear as user-facing labels
        # (they may appear in backward-compat dict.get() fallbacks, which is OK)
        label_patterns = [
            r'st\.\w+\([^)]*"[^"]*Option A[^"]*"',
            r'st\.\w+\([^)]*"[^"]*Option C[^"]*"',
            r'st\.metric\([^)]*"[^"]*Option A[^"]*"',
            r'st\.metric\([^)]*"[^"]*Option C[^"]*"',
            r'st\.subheader\([^)]*"[^"]*Option A[^"]*"',
            r'st\.subheader\([^)]*"[^"]*Option C[^"]*"',
        ]
        for pattern in label_patterns:
            matches = re.findall(pattern, source)
            assert not matches, f"Found old naming in UI label: {matches}"

    def test_no_v1_v2_in_streamlit_labels(self) -> None:
        app_path = ROOT / "app" / "streamlit_app.py"
        source = app_path.read_text(encoding="utf-8")
        # "V1" / "V2" should not appear as user-facing labels
        label_patterns = [
            r'st\.metric\([^)]*"[^"]*\bV1\b[^"]*"',
            r'st\.metric\([^)]*"[^"]*\bV2\b[^"]*"',
        ]
        for pattern in label_patterns:
            matches = re.findall(pattern, source)
            assert not matches, f"Found old V1/V2 in UI label: {matches}"
