"""Thread-safe telemetry logger for NL-to-SQL predictions."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """Thread-safe persistent logger for prediction events and user feedback."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        if log_path is None:
            root = Path(__file__).resolve().parents[1]
            self.log_path = root / "artifacts" / "telemetry.jsonl"
        else:
            self.log_path = Path(log_path)
        self.lock = threading.Lock()

    def log_prediction(
        self,
        question: str,
        result: dict[str, Any],
        duration_ms: float = 0.0,
    ) -> None:
        """Log a prediction event with all metadata."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "prediction",
            "question": question,
            "duration_ms": duration_ms,
            "status": result.get("status"),
            "source_model": result.get("source_model"),
            "intent": result.get("intent"),
            "sql": result.get("sql"),
            "confidence": result.get("confidence"),
            "calibrated_confidence": result.get("calibrated_confidence"),
            "abstain": result.get("abstain"),
            "abstention_reason": result.get("abstention_reason"),
            "needs_clarification": result.get("needs_clarification"),
            "schema_drift_flags": result.get("schema_drift_flags", []),
            "bundle_id": (result.get("debug") or {}).get("bundle_id", ""),
            "runtime_source": (result.get("debug") or {}).get("runtime_source", ""),
        }
        self._write_entry(entry)

    def log_feedback(
        self,
        question: str,
        sql: str,
        is_correct: bool,
        comments: str | None = None,
    ) -> None:
        """Log user feedback for auditability."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "feedback",
            "question": question,
            "sql": sql,
            "is_correct": is_correct,
            "comments": comments,
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Thread-safe append to JSONL log file."""
        try:
            with self.lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error("Failed to write telemetry log entry: %s", exc)
