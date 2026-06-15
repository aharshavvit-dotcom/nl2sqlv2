"""Database connector layer for NL2SQL.

Provides unified schema reading and query execution for SQLite and PostgreSQL.
"""

from .connection_config import DatabaseConnectionConfig, safe_config_summary
from .connector_base import DatabaseConnector
from .dialect import SUPPORTED_DIALECTS, get_sqlglot_dialect
from .schema_reader import read_database_schema, schema_dict_to_graph
from .sqlite_connector import SQLiteConnector

__all__ = [
    "DatabaseConnectionConfig",
    "DatabaseConnector",
    "SQLiteConnector",
    "read_database_schema",
    "safe_config_summary",
    "schema_dict_to_graph",
    "get_sqlglot_dialect",
    "SUPPORTED_DIALECTS",
]

try:
    from .postgres_connector import PostgresConnector
    __all__.append("PostgresConnector")
except ImportError:
    pass
