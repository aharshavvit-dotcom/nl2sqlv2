from __future__ import annotations

from uuid import uuid4

from ir.query_ir_models import IRDimension, IRFilter, IRMetric, QueryIR

from .join_policy import JoinPolicy
from .schema_profile import SchemaProfile


class DirectQueryIRBuilder:
    def __init__(self, schema_profile: SchemaProfile):
        self.schema_profile = schema_profile

    def build_show_records(
        self,
        table: str,
        limit: int = 100,
        selected_columns: list[str] | None = None,
        question: str = "",
    ) -> QueryIR:
        columns = selected_columns or self.schema_profile.safe_select_columns(table)
        warnings = [] if columns else [f"No safe selectable columns found for table {table}."]
        dimensions = [
            IRDimension(
                name=column,
                table=table,
                column=column,
                expression=f"{table}.{column}",
                alias=column,
                source_slot="generic_direct_planner",
                confidence=1.0,
            )
            for column in columns
        ]
        return self._query_ir(
            question=question,
            intent="show_records",
            template_id="show_records",
            table=table,
            dimensions=dimensions,
            warnings=warnings,
            limit=limit,
            select_mode="records",
            metadata_extra={"safe_selected_columns": columns},
        )

    def build_count_records(self, table: str, question: str = "") -> QueryIR:
        metric = IRMetric(
            name="record_count",
            aggregation="COUNT",
            table=table,
            column="*",
            expression="*",
            alias="record_count",
            source_slot="generic_direct_planner",
            confidence=1.0,
        )
        return self._query_ir(
            question=question,
            intent="count_records",
            template_id="count_records",
            table=table,
            metrics=[metric],
            limit=100,
            select_mode="count",
        )

    def build_simple_filter(
        self,
        table: str,
        filter_column: str,
        filter_operator: str,
        filter_value: str,
        limit: int = 100,
        question: str = "",
    ) -> QueryIR:
        columns = self.schema_profile.safe_select_columns(table)
        warnings = [] if columns else [f"No safe selectable columns found for table {table}."]
        dimensions = [
            IRDimension(
                name=column,
                table=table,
                column=column,
                expression=f"{table}.{column}",
                alias=column,
                source_slot="generic_direct_planner",
                confidence=1.0,
            )
            for column in columns
        ]
        ir_filter = IRFilter(
            name=filter_column,
            table=table,
            column=filter_column,
            expression=f"{table}.{filter_column}",
            operator=filter_operator,  # type: ignore[arg-type]
            value=filter_value,
            value_type="string",
            raw_text=f"{filter_column} {filter_operator} {filter_value}",
            confidence=1.0,
        )
        return self._query_ir(
            question=question,
            intent="simple_filter",
            template_id="simple_filter",
            table=table,
            dimensions=dimensions,
            filters=[ir_filter],
            warnings=warnings,
            limit=limit,
            select_mode="records",
            metadata_extra={"safe_selected_columns": columns},
        )

    def _query_ir(
        self,
        question: str,
        intent: str,
        template_id: str,
        table: str,
        metrics: list[IRMetric] | None = None,
        dimensions: list[IRDimension] | None = None,
        filters: list[IRFilter] | None = None,
        warnings: list[str] | None = None,
        limit: int = 100,
        select_mode: str = "records",
        metadata_extra: dict | None = None,
    ) -> QueryIR:
        metadata = {
            "source": "generic_direct_planner",
            "join_policy": JoinPolicy.NONE.value,
            "force_quoted_identifiers": True,
            "validation_context": {
                "schema_context": {
                    "tables": {
                        table_name: {
                            "columns": {column["name"]: column for column in self.schema_profile.get_columns(table_name)}
                        }
                        for table_name in self.schema_profile.get_tables()
                    },
                    "dialect": self.schema_profile.dialect,
                }
            },
        }
        metadata.update(metadata_extra or {})
        return QueryIR(
            query_ir_id=f"qir_{uuid4().hex}",
            question=question,
            normalized_question=" ".join(question.lower().split()),
            intent=intent,
            template_id=template_id,
            dialect=self.schema_profile.dialect,
            base_table=table,
            required_tables=[table],
            metrics=metrics or [],
            dimensions=dimensions or [],
            filters=filters or [],
            date_filters=[],
            joins=[],
            group_by=[],
            order_by=[],
            limit=limit,
            select_mode=select_mode,  # type: ignore[arg-type]
            warnings=warnings or [],
            metadata=metadata,
        )
