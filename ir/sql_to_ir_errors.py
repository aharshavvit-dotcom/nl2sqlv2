from __future__ import annotations


class SQLToIRError(Exception):
    def __init__(self, reason: str, message: str, sql: str | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.sql = sql

    def to_dict(self) -> dict[str, str | None]:
        return {
            "reason": self.reason,
            "message": self.message,
            "sql": self.sql,
        }


class UnsupportedSQLPattern(SQLToIRError):
    pass


class SQLParseFailure(SQLToIRError):
    pass


class SchemaResolutionFailure(SQLToIRError):
    pass


class IRConstructionFailure(SQLToIRError):
    pass


PARSE_ERROR = "parse_error"
NON_SELECT = "non_select"
NESTED_QUERY = "nested_query"
SET_OPERATION = "set_operation"
WINDOW_FUNCTION = "window_function"
UNSUPPORTED_HAVING = "unsupported_having"
UNSUPPORTED_CASE = "unsupported_case"
UNSUPPORTED_EXPRESSION = "unsupported_expression"
MISSING_BASE_TABLE = "missing_base_table"
MISSING_METRIC = "missing_metric"
MISSING_DIMENSION = "missing_dimension"
MISSING_JOIN = "missing_join"
UNKNOWN_SCHEMA_REFERENCE = "unknown_schema_reference"

