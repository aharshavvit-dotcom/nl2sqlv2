# Production Runtime Policy

Production runtime must load a validated current bundle through `ModelBundleLoader.load_current`.

The bundle manifest is the owner for:

- Routing policy
- Retrieval artifact paths
- Neural artifact paths
- Quality-gate status
- Runtime readiness proof

UI or debug inputs must not override production routing or bundle path.

Prediction cache policy:

- SQLite cache only.
- Raw questions are not stored.
- Cache keys include bundle/checkpoint/retrieval/routing identity when provided.
- Production cache is disabled when tenant or security context is missing.
- Failed, invalid, unsafe, or partial predictions are not cached.

Telemetry policy:

- Raw questions, raw SQL, result values, and feedback text are disabled by default.
- Recursive sanitization is applied before optional raw logging.
