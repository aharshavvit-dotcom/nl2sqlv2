from __future__ import annotations

from typing import Any

from ir.sql_to_ir_converter import SQLToIRConverter


class CorrectionParser:
    def __init__(self, dialect: str = "sqlite", max_limit: int = 1000):
        self.dialect = dialect
        self.max_limit = max_limit

    def corrected_sql_to_query_ir(
        self,
        question: str,
        corrected_sql: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        converter = SQLToIRConverter(dialect=self.dialect, max_limit=self.max_limit)
        return converter.convert(question=question, sql=corrected_sql, schema=schema)
