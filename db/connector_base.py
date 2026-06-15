"""Abstract base class for database connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatabaseConnector(ABC):
    """Interface that all database connectors must implement."""

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Test the connection and return ``(success, message)``."""

    @abstractmethod
    def read_schema(self) -> dict[str, Any]:
        """Return the database schema in normalized dict format."""

    @abstractmethod
    def execute_readonly(self, sql: str, limit: int | None = None) -> dict[str, Any]:
        """Execute a read-only query and return ``{"columns": [...], "rows": [...]}``."""

    @abstractmethod
    def get_dialect(self) -> str:
        """Return the SQL dialect name (``'sqlite'`` or ``'postgres'``)."""
