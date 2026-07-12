# QueryIR v2 Foundation Specification

Phase 2A introduces a typed QueryIR v2 foundation while preserving the v1
runtime path. It does not add advanced SQL rendering, train new heads, change
model output, or change production routing.

## Version contract

Every v2 query object contains:

```json
{
  "query_ir_version": "2.0"
}
```

The version-aware loader rejects unknown versions. Legacy payloads without a
version are treated as v1 only inside `load_query_ir`.

## v2 model surface

The foundation lives in `ir/query_ir_v2_models.py` and defines:

- `QueryNode`
- `SelectItem`
- `FromItem`
- `JoinNode`
- `OrderByItem`
- `SetOperationNode`
- discriminated expression union
- discriminated predicate union
- `WindowSpecification`
- `CapabilityMetadata`
- `ConfidenceMetadata`

Expression discriminator: `expression_type`.

Supported expression variants are `COLUMN`, `LITERAL`, `FUNCTION`,
`AGGREGATION`, `BINARY_OPERATION`, `UNARY_OPERATION`, `BOOLEAN_OPERATION`,
`CASE_EXPRESSION`, `SUBQUERY`, and `WINDOW_EXPRESSION`.

Advanced variants may validate as v2 data but are not rendered in Phase 2A.
The v2 renderer adapter raises `unsupported_v2_rendering_capability` for
unsupported renderer capabilities.

## Migration mapping

`ir/query_ir_migration.py` implements deterministic migration.

| v1 field | v2 mapping |
| --- | --- |
| `query_ir_id`, `question`, `normalized_question`, `intent`, `template_id`, `dialect` | Same top-level fields on `QueryNode`. |
| `base_table` | `from_item = FromItem(from_type="TABLE", table=base_table)`. |
| `required_tables` | `QueryNode.required_tables`. |
| `dimensions` | `SelectItem(role="dimension", expression=COLUMN, legacy_v1=...)`. |
| `metrics` | `SelectItem(role="metric" or "count", expression=AGGREGATION(...), legacy_v1=...)`. |
| Product/binary metric expressions | `AGGREGATION(argument=BINARY_OPERATION(...))`. |
| `filters` | Phase 2B predicate nodes such as `COMPARISON_PREDICATE`, `IN_LITERAL_PREDICATE`, and `BOOLEAN_PREDICATE`. |
| `date_filters` | `DateFilterNode` preserving date grain/range fields. |
| `joins` | `JoinNode` with right table and equality predicate. |
| `group_by` | v2 expressions, preserving date-grain function placeholders. |
| `order_by` | `OrderByItem`. |
| `limit` | `QueryNode.limit`. |
| `select_mode`, `warnings`, `metadata` | Same top-level fields; original v1 payload is retained under `metadata.v1_payload` for lossless compatibility. |

`convert_v2_to_v1` supports the subset representable by QueryIR v1. Unsupported
advanced nodes return a typed `QueryIRCompatibilityError`.

## Compatibility matrix

| Capability | v2 validates | v2-to-v1 converts | Renderer support in Phase 2A |
| --- | --- | --- | --- |
| Simple projection | Yes | Yes | Existing v1 renderer |
| Aggregations | Yes | Yes | Existing v1 renderer |
| Binary metric expression | Yes | Yes | Existing v1 behavior preserved |
| AND filter list | Yes | Yes | Existing v1 renderer |
| Joins | Yes | Yes | Existing v1 renderer |
| Date range/grain placeholders | Yes | Yes | Existing v1 renderer |
| OR predicates | Yes | No v1 conversion | Phase 2B native renderer only with diagnostic/test flag |
| CASE | Yes | No | `unsupported_v2_rendering_capability` |
| Subquery | Yes | No | `unsupported_v2_rendering_capability` |
| Window expression | Yes | No | `unsupported_v2_rendering_capability` |
| Set operation | Yes | No | `unsupported_v2_rendering_capability` |
| HAVING | Not rendered | No | Non-goal |

## Validation

`QueryIRV2Validator` validates discriminated unions, recursive depth limits,
unique select aliases, reference shape, join/order/limit constraints,
capability metadata consistency, mutation-query rejection, and optional current
renderer support.

The default recursive depth limit is configurable and exists to protect loaders
from malicious or malformed recursive payloads.

## Serialization and fingerprinting

`ir/query_ir_v2_serialization.py` provides deterministic JSON serialization,
canonical nested key ordering, deserialization, and SHA-256 fingerprinting.
Fingerprints include `query_ir_version`.

## Phase 2A runtime policy

Production defaults remain v1:

- Model output version: `1`
- Runtime preferred version: `1`
- Supported versions declared in future bundles: `["1", "2.0"]`

The v2 compatibility adapter converts v2 back to v1 and delegates to
`IRToSQLRenderer`; it does not rewrite the renderer.

## Known limitations

- Advanced SQL nodes are type foundations only.
- No HAVING, CASE rendering, subquery rendering, window rendering, or
  set-operation rendering. OR rendering is Phase 2B diagnostic/test-only and
  remains disabled for production routing.
- QueryIR v2 does not change the active neural architecture, routing, or label
  encoder contract.
- v2-to-v1 conversion is only guaranteed for the v1-compatible subset.

## Phase 2B recommendation

Start Phase 2B with one advanced construct at a time. The safest next step is
to choose a single capability, add renderer support behind explicit capability
checks, add parity and negative tests, and keep production routing on v1 until
the new renderer path has equivalent regression coverage.
