# Privacy And Retention Policy

Telemetry defaults:

- Raw question: disabled
- Raw SQL: disabled
- Result values: disabled
- Feedback text: disabled

Logged production metadata should be limited to hashes and operational fields such as bundle ID, route, intent, latency, schema fingerprint, validation result, abstention/clarification status, tenant/environment when approved, and sanitized error category.

Redaction includes:

- Email
- Phone
- Luhn-valid credit cards
- IPv4/IPv6
- UUID
- PAN-like IDs
- Aadhaar-like IDs
- JWTs
- Bearer/access tokens
- Password/API-key/secret assignments
- Connection strings
- Private keys

Prediction cache privacy:

- Raw questions are hashed.
- Filter literal values are redacted in cached QueryIR.
- Production cache requires tenant and security-context identity.

Retention and rotation:

- Telemetry rotates at configured max size with limited backups.
- Cache supports TTL and maximum entries.
- Additional max database size and corruption recovery policies remain deferred.
