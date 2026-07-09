"""Unit and privacy compliance tests for the TelemetryLogger sanitization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from inference.telemetry_logger import TelemetryLogger


@pytest.fixture
def telemetry_logger(tmp_path):
    log_file = tmp_path / "telemetry.jsonl"
    return TelemetryLogger(log_path=log_file)


def test_telemetry_default_excludes_raw_content(telemetry_logger):
    question = "Please email target@example.com or call 555-019-9000 about credit card 4111111111111111"
    result = {
        "status": "completed",
        "source_model": "neural",
        "intent": "show_records",
        "sql": "SELECT * FROM users",
        "confidence": 0.9,
    }
    
    # Defaults do NOT write raw question or SQL text
    telemetry_logger.log_prediction(question, result)
    
    log_content = telemetry_logger.log_path.read_text(encoding="utf-8")
    log_entry = json.loads(log_content.strip())
    
    assert "question_hash" in log_entry
    assert "raw_question" not in log_entry
    assert "raw_sql" not in log_entry


def test_telemetry_pii_sanitization_rules(telemetry_logger):
    # Enable raw logs to inspect PII redactor outputs
    os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_QUESTION"] = "1"
    os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_SQL"] = "1"
    
    try:
        # Valid Luhn card (VISA test card)
        visa_card = "4111 1111 1111 1111"
        # Non-Luhn card (arbitrary number sequence)
        non_luhn = "1234-5678-9012-3456"
        
        question = f"My email is support@acme.com, phone 1-206-555-0100, card {visa_card}, order {non_luhn}."
        result = {
            "status": "completed",
            "source_model": "neural",
            "sql": "SELECT * FROM users WHERE apikey = 'sk-live-123456789abc'",
        }
        
        telemetry_logger.log_prediction(question, result)
        
        log_content = telemetry_logger.log_path.read_text(encoding="utf-8")
        log_entry = json.loads(log_content.strip())
        
        raw_q = log_entry["raw_question"]
        raw_sql = log_entry["raw_sql"]
        
        assert "support@acme.com" not in raw_q
        assert "[EMAIL]" in raw_q
        
        assert "1-206-555-0100" not in raw_q
        assert "[PHONE]" in raw_q
        
        # VISA card should be redacted because it passes Luhn check
        assert visa_card not in raw_q
        assert "[CARD]" in raw_q
        
        # Order number/Non-Luhn card should NOT be redacted as card
        assert non_luhn in raw_q
        
        # API keys/Secrets in SQL must be redacted
        assert "sk-live-123456789abc" not in raw_sql
        assert "[SECRET]" in raw_sql
        
    finally:
        del os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_QUESTION"]
        del os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_SQL"]


def test_recursive_payload_sanitizer(telemetry_logger):
    payload = {
        "user_profile": {
            "email": "user@gmail.com",
            "phones": ["+1-555-019-2831", "just a string"]
        },
        "query_history": [
            "show cards matching 4111-1111-1111-1111",
        ]
    }
    
    sanitized = telemetry_logger.sanitize_payload(payload)
    
    assert sanitized["user_profile"]["email"] == "[EMAIL]"
    assert sanitized["user_profile"]["phones"][0] == "[PHONE]"
    assert sanitized["user_profile"]["phones"][1] == "just a string"
    assert "[CARD]" in sanitized["query_history"][0]


def test_telemetry_logger_permissions_and_rotation(telemetry_logger):
    # Setup small rotation bytes
    telemetry_logger.max_bytes = 20
    
    # Write entries to trigger rotation
    telemetry_logger._write_entry({"data": "short"})
    telemetry_logger._write_entry({"data": "trigger_rotation"})
    telemetry_logger._write_entry({"data": "three"})
    
    # Check that rotation file exists
    assert telemetry_logger.log_path.exists()
    assert telemetry_logger.log_path.with_name(f"{telemetry_logger.log_path.name}.1").exists()
    
    # Permissions check (only on Unix/MacOS, skip on Windows)
    if os.name == "posix":
        mode = telemetry_logger.log_path.stat().st_mode
        assert oct(mode & 0o777) == "0o600"

