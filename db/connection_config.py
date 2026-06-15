"""Database connection configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatabaseConnectionConfig:
    """Configuration for a database connection.

    Supports SQLite and PostgreSQL.  Use ``safe_config_summary`` to
    obtain a dict representation that masks sensitive fields.
    """

    db_type: str = "sqlite"
    sqlite_path: str | None = None

    # PostgreSQL connection parameters
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    sslmode: str | None = None
    schema_name: str | None = None

    def __post_init__(self) -> None:
        self.db_type = (self.db_type or "sqlite").lower()
        if self.db_type not in ("sqlite", "postgres", "postgresql"):
            raise ValueError(f"Unsupported db_type: {self.db_type!r}.  Use 'sqlite' or 'postgres'.")
        if self.db_type in ("postgres", "postgresql"):
            self.db_type = "postgres"
        if self.port is not None:
            self.port = int(self.port)
        if self.db_type == "postgres" and self.schema_name is None:
            self.schema_name = "public"

    @property
    def dialect(self) -> str:
        return self.db_type

    def sqlalchemy_url(self) -> str:
        """Build a SQLAlchemy connection URL.

        For PostgreSQL the password is embedded; callers should never
        log the return value of this method.
        """
        if self.db_type == "sqlite":
            from pathlib import Path
            path = Path(self.sqlite_path or "").resolve()
            return f"sqlite:///{path.as_posix()}"
        # PostgreSQL
        user = self.username or ""
        pwd = self.password or ""
        host = self.host or "localhost"
        port = self.port or 5432
        db = self.database or ""
        base = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        params = []
        if self.sslmode:
            params.append(f"sslmode={self.sslmode}")
        if params:
            base += "?" + "&".join(params)
        return base


def safe_config_summary(config: DatabaseConnectionConfig) -> dict[str, Any]:
    """Return a dict summary of *config* with the password masked."""
    summary: dict[str, Any] = {"db_type": config.db_type}
    if config.db_type == "sqlite":
        summary["sqlite_path"] = config.sqlite_path
    else:
        summary["host"] = config.host
        summary["port"] = config.port
        summary["database"] = config.database
        summary["username"] = config.username
        summary["password"] = "***" if config.password else None
        summary["sslmode"] = config.sslmode
        summary["schema_name"] = config.schema_name
    return summary
