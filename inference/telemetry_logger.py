"""Thread-safe, privacy-hardened telemetry logger for NL-to-SQL predictions.

Enforces structured metadata logging by default, disables raw questions/SQL text
unless explicitly permitted, and recursive PII sanitization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def luhn_check(num_str: str) -> bool:
    """Validate numeric string using Luhn algorithm (checksum verification)."""
    digits = [int(d) for d in num_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        double = d * 2
        total += double if double < 10 else double - 9
    return total % 10 == 0


class TelemetryLogger:
    """Thread-safe persistent logger for prediction events and user feedback."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        if log_path is None:
            root = Path(__file__).resolve().parents[1]
            self.log_path = root / "artifacts" / "telemetry.jsonl"
        else:
            self.log_path = Path(log_path)
            
        self.lock = threading.Lock()
        self.max_bytes = 5 * 1024 * 1024  # 5 MB log rotation threshold
        self.backup_count = 3
        self.failure_counter = 0

    def _should_log_raw_question(self) -> bool:
        return os.getenv("NL2SQL_TELEMETRY_INCLUDE_RAW_QUESTION", "").strip().lower() in {"1", "true", "yes"}

    def _should_log_raw_sql(self) -> bool:
        return os.getenv("NL2SQL_TELEMETRY_INCLUDE_RAW_SQL", "").strip().lower() in {"1", "true", "yes"}

    def _should_log_feedback_comments(self) -> bool:
        return os.getenv("NL2SQL_TELEMETRY_INCLUDE_FEEDBACK_COMMENTS", "").strip().lower() in {"1", "true", "yes"}

    def sanitize_payload(self, data: Any) -> Any:
        """Recursively scrub sensitive information/PII from dictionaries, lists, or strings."""
        if isinstance(data, dict):
            return {k: self.sanitize_payload(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_payload(x) for x in data]
        elif isinstance(data, str):
            return self.redact_pii_string(data)
        return data

    def redact_pii_string(self, text: str) -> str:
        """Identify and redact emails, phone numbers, credit cards, UUIDs, IPs, and tokens."""
        # 1. Emails
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", text)

        # 2. Credit Cards (Processed first to avoid collision with phone numbers)
        card_candidates = re.findall(r"\b(?:\d[-.\s]?){13,19}\b", text)
        for cand in card_candidates:
            clean_cand = re.sub(r"\D", "", cand)
            if luhn_check(clean_cand):
                text = text.replace(cand, "[CARD]")

        # 3. Phone Numbers (Refined pattern to avoid 16-digit blocks)
        # Matches country code patterns (e.g. +1-206-555-0100) or standard national (e.g. 555-019-9000)
        text = re.sub(r"\+?\b\d{1,2}[-.\s]\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]", text)
        text = re.sub(r"\+?\b\d{1,3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]", text)

        # 4. UUIDs
        text = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "[UUID]", text)

        # 5. IP Addresses (IPv4 and IPv6 patterns)
        text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", text)
        text = re.sub(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", "[IP]", text)

        # 6. API Keys / Secret assignments (Redact credentials/keys matching keywords)
        text = re.sub(
            r"(?i)(api_key|apikey|secret|token|password|conn_str|connection_string)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{8,})[\"']?",
            r"\1: [SECRET]",
            text,
        )
        return text

    def log_prediction(
        self,
        question: str,
        result: dict[str, Any],
        duration_ms: float = 0.0,
    ) -> None:
        """Log structured prediction metadata, redacting raw text unless permitted."""
        q_normalized = re.sub(r"\s+", " ", question.strip().lower())
        q_hash = hashlib.sha256(q_normalized.encode("utf-8")).hexdigest()

        # Build clean structured entry
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "prediction",
            "question_hash": q_hash,
            "duration_ms": duration_ms,
            "status": result.get("status"),
            "source_model": result.get("source_model"),
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "calibrated_confidence": result.get("calibrated_confidence"),
            "abstain": result.get("abstain"),
            "abstention_reason": result.get("abstention_reason"),
            "needs_clarification": result.get("needs_clarification"),
            "schema_drift_flags": result.get("schema_drift_flags", []),
            "bundle_id": (result.get("debug") or {}).get("bundle_id", ""),
            "runtime_source": (result.get("debug") or {}).get("runtime_source", ""),
        }

        # Conditionally append raw information with redaction engine
        if self._should_log_raw_question():
            entry["raw_question"] = self.redact_pii_string(question)
            
        if self._should_log_raw_sql() and result.get("sql"):
            entry["raw_sql"] = self.redact_pii_string(result["sql"])

        self._write_entry(entry)

    def log_feedback(
        self,
        question: str,
        sql: str,
        is_correct: bool,
        comments: str | None = None,
    ) -> None:
        """Log user feedback safely."""
        q_normalized = re.sub(r"\s+", " ", question.strip().lower())
        q_hash = hashlib.sha256(q_normalized.encode("utf-8")).hexdigest()

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "feedback",
            "question_hash": q_hash,
            "is_correct": is_correct,
        }

        if self._should_log_raw_question():
            entry["raw_question"] = self.redact_pii_string(question)
        if self._should_log_raw_sql():
            entry["raw_sql"] = self.redact_pii_string(sql)
        if self._should_log_feedback_comments() and comments:
            entry["comments"] = self.redact_pii_string(comments)

        self._write_entry(entry)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Thread-safe append with log rotation and locked file permissions."""
        try:
            with self.lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check log rotation
                if self.log_path.exists() and self.log_path.stat().st_size >= self.max_bytes:
                    self._rotate_logs()

                # Open and ensure locked down permissions (owner read-write only)
                fd = os.open(str(self.log_path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    os.close(fd)
                    raise e
        except Exception as exc:
            self.failure_counter += 1
            logger.warning("Telemetry write failed: %s. Application prediction continues.", exc)

    def _rotate_logs(self) -> None:
        """Perform sequential log rotation up to backup_count limits."""
        for i in range(self.backup_count - 1, 0, -1):
            src = self.log_path.with_name(f"{self.log_path.name}.{i}")
            dest = self.log_path.with_name(f"{self.log_path.name}.{i+1}")
            if src.exists():
                if dest.exists():
                    dest.unlink()
                src.rename(dest)
        dest = self.log_path.with_name(f"{self.log_path.name}.1")
        if dest.exists():
            dest.unlink()
        self.log_path.rename(dest)
