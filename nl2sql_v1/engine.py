from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .renderer import render_sql
from .retriever import RetrievalResult, TfidfRetriever
from .schema import SchemaGraph
from .schema_matcher import MatchedDimension, MatchedFilter, MatchedMetric, SchemaMatcher
from .slot_extractor import ExtractedSlots, SlotExtractor
from .template_adapter import RenderPlan, TemplateAdapter
from .validator import validate_select_sql


@dataclass(frozen=True)
class GenerationResult:
    question: str
    sql: str
    template_id: str
    retrieval: RetrievalResult
    retrieved_examples: list[RetrievalResult]
    slots: ExtractedSlots
    metric: MatchedMetric
    dimension: MatchedDimension | None
    filters: list[MatchedFilter]
    plan: RenderPlan


class NL2SQLEngine:
    def __init__(self, retriever: TfidfRetriever, templates_path: str | Path, synonyms_path: str | Path):
        self.retriever = retriever
        self.matcher = SchemaMatcher.from_yaml(synonyms_path)
        self.extractor = SlotExtractor(self.matcher.catalog)
        self.adapter = TemplateAdapter.from_yaml(templates_path)

    def generate(self, question: str, schema: SchemaGraph) -> GenerationResult:
        retrieved_examples = self.retriever.query(question, top_k=5)
        retrieval = retrieved_examples[0]
        slots = self.extractor.extract(question, fallback=retrieval.example)
        matched = self.matcher.match(slots, schema)
        template_id = slots.template_id or retrieval.template_id
        plan = self.adapter.build_plan(
            template_id=template_id,
            schema=schema,
            metric=matched.metric,
            dimension=matched.dimension,
            filters=matched.filters,
            slots=slots,
        )
        sql = render_sql(plan.template, plan.context)
        validation = validate_select_sql(sql, schema)
        if not validation.ok:
            raise ValueError(validation.message)
        return GenerationResult(
            question=question,
            sql=sql,
            template_id=plan.template_id,
            retrieval=retrieval,
            retrieved_examples=retrieved_examples,
            slots=slots,
            metric=matched.metric,
            dimension=matched.dimension,
            filters=matched.filters,
            plan=plan,
        )
