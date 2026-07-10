# Training Data Lineage

Every supported corpus row now carries:

- `source_dataset`
- `source_dataset_version`
- `source_split`
- `source_example_id`
- `database_id`
- `internal_split`
- `eligible_for_training`
- `content_hash`

Rules enforced in `dataset_training/split_manager.py`:

- Manifest databases cannot overlap across splits.
- Manifest databases cannot be unknown.
- Dataset databases cannot be absent from a manifest and silently default to train.
- Required production split IDs must be present when applying a manifest.
- Dataset hashes must match.
- Source `test`, `dev`, or `validation` records cannot be assigned to internal train.

Leakage owner: `dataset_training/leakage_checker.py`.

Strict blockers include database overlap, exact non-generic question leakage, SQL leakage, canonical QueryIR leakage, SQL AST leakage, near duplicates, parent-child transitive leakage, and schema-family unseen leakage.

Generic template overlap, such as short "list customers" style phrases across unrelated schemas, is reported but not strict-blocking.
