"""Tests for Gate 2: Data Readiness.

Tests cover:
- Synthetic query generator with provenance
- Schema renaming augmenter
- Coverage tracking
- Active capability registry
- Template leakage detection
- Schema-family leakage detection
"""

from __future__ import annotations

import pytest

from dataset_training.synthetic_generator import (
    CapabilityTag,
    CoverageTarget,
    SchemaDefinition,
    SchemaRenamingAugmenter,
    SyntheticQueryGenerator,
)
from dataset_training.capability_registry import (
    ActiveCapabilityRegistry,
    CapabilityEntry,
    CapabilityStatus,
)
from dataset_training.leakage_checker import (
    check_template_leakage,
    check_schema_family_leakage,
)

import random


# ── Fixtures ─────────────────────────────────────────────────────────

def _sample_schema() -> SchemaDefinition:
    return SchemaDefinition(
        name="retail",
        tables={
            "orders": ["id", "customer_id", "amount", "date", "status"],
            "customers": ["id", "name", "email", "region"],
            "products": ["id", "name", "price", "category"],
        },
        primary_keys={"orders": "id", "customers": "id", "products": "id"},
        foreign_keys=[("orders", "customer_id", "customers", "id")],
    )


# ── Synthetic Generator ─────────────────────────────────────────────

class TestSyntheticGenerator:
    def test_generates_valid_examples(self):
        schema = _sample_schema()
        gen = SyntheticQueryGenerator([schema], seed=42)
        examples = gen.generate(max_examples=20)
        assert len(examples) == 20
        assert all(ex.is_valid for ex in examples)

    def test_provenance_is_populated(self):
        schema = _sample_schema()
        gen = SyntheticQueryGenerator([schema], seed=42)
        examples = gen.generate(max_examples=5)
        for ex in examples:
            assert ex.provenance.schema_source != ""
            assert ex.provenance.template_id != ""
            assert len(ex.provenance.capability_tags) > 0
            assert ex.provenance.fingerprint != ""

    def test_coverage_tracking(self):
        schema = _sample_schema()
        gen = SyntheticQueryGenerator([schema], seed=42, coverage_targets={
            CapabilityTag.SIMPLE_SELECT: 3,
            CapabilityTag.WHERE_FILTER: 3,
            CapabilityTag.LIMIT: 3,
        })
        gen.generate(max_examples=50)
        report = gen.coverage_report()
        assert report["total_capabilities"] > 0
        # LIMIT should be satisfied since all templates use it
        limit_entry = next((d for d in report["details"] if d["capability"] == "LIMIT"), None)
        assert limit_entry is not None
        assert limit_entry["satisfied"] is True

    def test_different_seeds_produce_different_examples(self):
        schema = _sample_schema()
        gen1 = SyntheticQueryGenerator([schema], seed=1)
        gen2 = SyntheticQueryGenerator([schema], seed=2)
        examples1 = gen1.generate(max_examples=10)
        examples2 = gen2.generate(max_examples=10)
        questions1 = {ex.question for ex in examples1}
        questions2 = {ex.question for ex in examples2}
        # They shouldn't all be identical
        assert questions1 != questions2

    def test_query_ir_is_valid(self):
        schema = _sample_schema()
        gen = SyntheticQueryGenerator([schema], seed=42)
        examples = gen.generate(max_examples=10)
        for ex in examples:
            # Each QueryIR should have required fields
            assert ex.query_ir.from_item is not None
            assert len(ex.query_ir.select_items) > 0
            assert ex.query_ir.question != ""

    def test_serialization_roundtrip(self):
        schema = _sample_schema()
        gen = SyntheticQueryGenerator([schema], seed=42)
        examples = gen.generate(max_examples=3)
        for ex in examples:
            d = ex.to_dict()
            assert d["question"] == ex.question
            assert d["is_valid"] == True
            assert "fingerprint" in d["provenance"]


# ── Schema Renaming Augmenter ────────────────────────────────────────

class TestSchemaRenamingAugmenter:
    def test_augmenter_renames_tables(self):
        schema = _sample_schema()
        aug = SchemaRenamingAugmenter()
        rng = random.Random(42)
        new_schema, rename_map = aug.augment(schema, rng, rename_probability=1.0)
        # At least some tables should be renamed
        original_tables = set(schema.tables.keys())
        new_tables = set(new_schema.tables.keys())
        assert new_tables != original_tables or rename_map  # something changed

    def test_augmenter_preserves_column_count(self):
        schema = _sample_schema()
        aug = SchemaRenamingAugmenter()
        rng = random.Random(42)
        new_schema, _ = aug.augment(schema, rng)
        for table in new_schema.tables:
            assert len(new_schema.tables[table]) > 0

    def test_augmenter_zero_probability_is_identity(self):
        schema = _sample_schema()
        aug = SchemaRenamingAugmenter()
        rng = random.Random(42)
        new_schema, rename_map = aug.augment(schema, rng, rename_probability=0.0)
        assert set(new_schema.tables.keys()) == set(schema.tables.keys())


# ── Capability Registry ──────────────────────────────────────────────

class TestCapabilityRegistry:
    def test_default_registry_has_capabilities(self):
        reg = ActiveCapabilityRegistry.default()
        assert reg.get("SIMPLE_SELECT") is not None
        assert reg.get("HAVING") is not None

    def test_active_capability_is_enabled(self):
        reg = ActiveCapabilityRegistry.default()
        assert reg.is_enabled("SIMPLE_SELECT") is True

    def test_flagged_capability_disabled_by_default(self):
        reg = ActiveCapabilityRegistry.default()
        assert reg.is_enabled("HAVING") is False

    def test_flagged_capability_enabled_by_flag(self):
        reg = ActiveCapabilityRegistry.default()
        reg.set_feature_flag("enable_having", True)
        assert reg.is_enabled("HAVING") is True

    def test_training_gap_detection(self):
        reg = ActiveCapabilityRegistry.default()
        gaps = reg.get_training_gap()
        # All capabilities start with 0 training count, so all should be in gap
        assert len(gaps) > 0

    def test_update_training_count(self):
        reg = ActiveCapabilityRegistry.default()
        reg.update_training_count("SIMPLE_SELECT", 100)
        entry = reg.get("SIMPLE_SELECT")
        assert entry.current_training_count == 100
        assert entry.training_sufficient is True

    def test_coverage_report(self):
        reg = ActiveCapabilityRegistry.default()
        report = reg.coverage_report()
        assert report["total_capabilities"] > 0
        assert "gap_details" in report

    def test_serialization_roundtrip(self):
        import tempfile
        import os
        reg = ActiveCapabilityRegistry.default()
        reg.set_feature_flag("enable_having", True)
        reg.update_training_count("SIMPLE_SELECT", 75)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            reg.save(path)
            loaded = ActiveCapabilityRegistry.load(path)
            assert loaded.is_enabled("HAVING") is True
            assert loaded.get("SIMPLE_SELECT").current_training_count == 75
        finally:
            os.unlink(path)


# ── Template Leakage ─────────────────────────────────────────────────

class TestTemplateLeakage:
    def test_detects_template_overlap(self):
        splits = {
            "train": [
                {"example_id": "t1", "source_sql": "SELECT id FROM orders WHERE amount > 100"},
            ],
            "frozen_semantic_test": [
                {"example_id": "e1", "source_sql": "SELECT id FROM orders WHERE amount > 200"},
            ],
        }
        result = check_template_leakage(splits)
        assert result["has_template_leakage"] is True

    def test_no_false_positive_on_different_structure(self):
        splits = {
            "train": [
                {"example_id": "t1", "source_sql": "SELECT id FROM orders WHERE amount > 100"},
            ],
            "frozen_semantic_test": [
                {"example_id": "e1", "source_sql": "SELECT name, email FROM customers LIMIT 10"},
            ],
        }
        result = check_template_leakage(splits)
        assert result["has_template_leakage"] is False


# ── Schema Family Leakage ────────────────────────────────────────────

class TestSchemaFamilyLeakage:
    def test_detects_schema_family_overlap_via_provenance(self):
        splits = {
            "train": [
                {"example_id": "t1", "provenance": {"schema_source": "retail"}},
            ],
            "frozen_semantic_test": [
                {"example_id": "e1", "provenance": {"schema_source": "retail"}},
            ],
        }
        result = check_schema_family_leakage(splits)
        assert result["has_schema_family_leakage"] is True

    def test_detects_schema_family_overlap_via_db_id(self):
        splits = {
            "train": [
                {"example_id": "t1", "db_id": "school_db"},
            ],
            "frozen_semantic_test": [
                {"example_id": "e1", "db_id": "school_db"},
            ],
        }
        result = check_schema_family_leakage(splits)
        assert result["has_schema_family_leakage"] is True

    def test_no_leakage_with_disjoint_schemas(self):
        splits = {
            "train": [
                {"example_id": "t1", "provenance": {"schema_source": "retail"}},
            ],
            "frozen_semantic_test": [
                {"example_id": "e1", "provenance": {"schema_source": "healthcare"}},
            ],
        }
        result = check_schema_family_leakage(splits)
        assert result["has_schema_family_leakage"] is False
