# QueryIR v1 Frozen Specification

Phase 2A freezes the current QueryIR v1 runtime contract without changing the
existing classes in `ir/query_ir_models.py`.

## Versioning

QueryIR v1 payloads do not contain `query_ir_version`. A payload without
`query_ir_version` is legacy v1 and must be interpreted through
`ir.query_ir_version_loader.load_query_ir`, not by silently guessing in new
v2-only code.

## Top-level QueryIR fields

| Field | Type | Default / rule |
| --- | --- | --- |
| `query_ir_id` | `str` | Required by the v1 Pydantic model. |
| `question` | `str` | Required. |
| `normalized_question` | `str` | Required. |
| `intent` | `str` | Required; validator supports executable intents only. |
| `template_id` | `str | None` | `None`. Renderer branches mostly on this field. |
| `dialect` | `str` | `"sqlite"`. Renderer also handles `"postgres"`. |
| `base_table` | `str | None` | `None`; executable validator requires it. |
| `required_tables` | `list[str]` | Empty list. |
| `metrics` | `list[IRMetric]` | Empty list. |
| `dimensions` | `list[IRDimension]` | Empty list. |
| `filters` | `list[IRFilter]` | Empty list. |
| `date_filters` | `list[IRDateFilter]` | Empty list. |
| `joins` | `list[IRJoin]` | Empty list. |
| `group_by` | `list[str]` | Empty list; strings are renderer expressions. |
| `order_by` | `list[IROrderBy]` | Empty list. |
| `limit` | `int` | `100`; validator requires `1 <= limit <= max_limit`. |
| `select_mode` | enum | `"records"`; values are `records`, `aggregate`, `trend`, `count`. |
| `warnings` | `list[str]` | Empty list. |
| `metadata` | `dict[str, Any]` | Empty dict; renderer reads validation context and quote flags. |

## Nested v1 models

`IRMetric`: `name`, `aggregation`, `table`, `column`, `expression`, `alias`,
`source_slot`, `confidence`. `aggregation` is a free string at model level; the
renderer uppercases it and supports current aggregate functions by convention.

`IRDimension`: `name`, `table`, `column`, `expression`, `alias`, `source_slot`,
`confidence`.

`IRFilter`: `name`, `table`, `column`, `expression`, `operator`, `value`,
`value_type`, `raw_text`, `confidence`. Operators are `equals`, `not_equals`,
`contains`, `in`, `not_in`, `greater_than`, `greater_equal`, `less_than`,
`less_equal`.

`IRDateFilter`: `date_table`, `date_column`, `date_expression`, `filter_type`,
`start_date`, `end_date`, `date_grain`, `raw_text`, `confidence`.
`filter_type` is `relative_range`, `absolute_range`, or `grain`.

`IRJoin`: `left_table`, `left_column`, `right_table`, `right_column`,
`join_type`, `condition`, `path_order`, `confidence`.

`IROrderBy`: `expression`, `alias`, `direction`, `source`. `direction` is
`ASC` or `DESC`; `source` is `metric`, `dimension`, `date`, `count`,
`explicit`, or `default`.

## Validation rules

The active validator is `IRValidator`:

- Requires executable intents from the current supported intent set.
- Requires `base_table` for executable QueryIR.
- Requires positive limit and caps it by validator `max_limit`.
- Requires metrics for metric templates and dimensions for by-dimension templates.
- Rejects unsafe direct table queries without safe select columns.
- Requires mapped date table and date column for date filters.
- Allows `*` only for `COUNT(*)`.
- Validates tables, columns, and join columns when a schema is supplied.
- Flags sensitive column references.

## Renderer assumptions

The production renderer is `IRToSQLRenderer`. It renders QueryIR v1 directly and
is still the production default in Phase 2A.

- Template-specific SELECT rendering is controlled by `template_id` and
  `select_mode`.
- Joins are sorted by `path_order`.
- Filters are joined with `AND`; OR, HAVING, CASE, subqueries, windows, and set
  operations are not rendered.
- Date grain rendering is limited to the current SQLite/Postgres branches.
- `LIMIT` is always emitted and clamped to renderer `max_limit`.

## Label encoder assumptions

`neural_ir/ir_label_encoder.py` consumes v1 dict payloads. It assumes:

- Intent labels are the current v1 intent list.
- Only the first metric, dimension, date filter, filter, and order-by item are
  encoded.
- Metric expression types are `none`, `column`, `count_star`, or
  `product_revenue_expression`.
- Model output remains QueryIR v1 in Phase 2A.

## Bundle serialization behavior

Existing bundles serialize manifests with v1 model-output assumptions. Phase
2A adds non-breaking manifest metadata:

```json
{
  "query_ir_versions_supported": ["1", "2.0"],
  "model_output_query_ir_version": "1",
  "runtime_preferred_query_ir_version": "1"
}
```

Legacy manifests missing these fields load with the same defaults.
