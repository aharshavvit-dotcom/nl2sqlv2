# Database Execution Policy

Required execution policy:

- SELECT only
- Single statement
- Read-only transaction
- Row limited
- Time limited
- Rollback protected
- Centrally validated before execution

SQLite owner: `db/sqlite_connector.py` and `execution/query_executor.py`.

PostgreSQL owner: `db/postgres_connector.py`.

Current readiness:

- SQLite has local tests.
- PostgreSQL has connector tests and policy checks, but real containerized integration coverage is not verified in this pass.

Production PostgreSQL readiness remains incomplete until a real integration profile verifies timeouts, read-only transactions, schema handling, RLS, role-specific access, JSON/NUMERIC/TIMESTAMPTZ behavior, and rollback semantics.
