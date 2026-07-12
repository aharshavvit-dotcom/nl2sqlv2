"""Tests for Gate 3: Architecture Readiness.

Tests cover:
- Feature flag infrastructure
- Schema graph construction and traversal
- Identifier decomposer
- Grammar state machine transitions and validation
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from training.feature_flags import (
    FeatureFlagRegistry,
    FlagDefinition,
    FlagType,
)
from training.schema_graph import (
    EdgeType,
    IdentifierDecomposer,
    NodeType,
    SchemaGraph,
)
from training.grammar_state_machine import (
    GrammarState,
    GrammarStateMachine,
    TokenCategory,
)


# ── Feature Flags ────────────────────────────────────────────────────

class TestFeatureFlags:
    def test_default_flags_exist(self):
        reg = FeatureFlagRegistry.default()
        assert reg.get_bool("enable_schema_graph") is False
        assert reg.get_bool("enable_grammar_decoder") is False
        assert reg.get_int("max_join_path_depth") == 4

    def test_override_takes_precedence(self):
        reg = FeatureFlagRegistry.default()
        reg.set_override("enable_schema_graph", True)
        assert reg.get_bool("enable_schema_graph") is True

    def test_env_var_override(self):
        reg = FeatureFlagRegistry.default()
        os.environ["NL2SQL_FLAG_ENABLE_SCHEMA_GRAPH"] = "true"
        try:
            assert reg.get_bool("enable_schema_graph") is True
        finally:
            del os.environ["NL2SQL_FLAG_ENABLE_SCHEMA_GRAPH"]

    def test_config_file_override(self):
        reg = FeatureFlagRegistry.default()
        reg.load_config_dict({"enable_schema_graph": True})
        assert reg.get_bool("enable_schema_graph") is True

    def test_override_beats_config(self):
        reg = FeatureFlagRegistry.default()
        reg.load_config_dict({"enable_schema_graph": True})
        reg.set_override("enable_schema_graph", False)
        assert reg.get_bool("enable_schema_graph") is False

    def test_unknown_flag_raises(self):
        reg = FeatureFlagRegistry.default()
        with pytest.raises(KeyError):
            reg.get("nonexistent_flag")

    def test_evaluation_log(self):
        reg = FeatureFlagRegistry.default()
        reg.get_bool("enable_schema_graph")
        log = reg.evaluation_log
        assert len(log) >= 1
        assert log[-1].flag_name == "enable_schema_graph"

    def test_all_flags_summary(self):
        reg = FeatureFlagRegistry.default()
        summary = reg.all_flags()
        assert "enable_schema_graph" in summary
        assert summary["enable_schema_graph"]["gate"] == 3

    def test_clear_override(self):
        reg = FeatureFlagRegistry.default()
        reg.set_override("enable_schema_graph", True)
        assert reg.get_bool("enable_schema_graph") is True
        reg.clear_override("enable_schema_graph")
        assert reg.get_bool("enable_schema_graph") is False


# ── Schema Graph ─────────────────────────────────────────────────────

class TestSchemaGraph:
    def _sample_graph(self) -> SchemaGraph:
        return SchemaGraph.from_schema_dict(
            {
                "orders": {"id": {"type": "int"}, "customer_id": {"type": "int"}, "amount": {"type": "float"}},
                "customers": {"id": {"type": "int"}, "name": {"type": "str"}},
            },
            primary_keys={"orders": "id", "customers": "id"},
            foreign_keys=[("orders", "customer_id", "customers", "id")],
        )

    def test_node_count(self):
        graph = self._sample_graph()
        # 2 tables + 5 columns = 7 nodes
        assert graph.num_nodes == 7

    def test_table_nodes(self):
        graph = self._sample_graph()
        tables = graph.get_table_nodes()
        assert len(tables) == 2
        names = {t.name for t in tables}
        assert names == {"orders", "customers"}

    def test_column_nodes_for_table(self):
        graph = self._sample_graph()
        cols = graph.get_column_nodes("orders")
        assert len(cols) == 3
        col_names = {c.name for c in cols}
        assert col_names == {"id", "customer_id", "amount"}

    def test_primary_key_node_feature(self):
        graph = self._sample_graph()
        pk_node = graph.get_node("col:orders.id")
        assert pk_node is not None
        assert pk_node.is_primary_key is True

    def test_foreign_key_edge_exists(self):
        graph = self._sample_graph()
        fk_edges = [e for e in graph.edges if e.edge_type == EdgeType.FOREIGN_KEY]
        assert len(fk_edges) == 1
        assert fk_edges[0].source == "col:orders.customer_id"
        assert fk_edges[0].target == "col:customers.id"

    def test_table_fk_edge_exists(self):
        graph = self._sample_graph()
        tfk_edges = [e for e in graph.edges if e.edge_type == EdgeType.TABLE_FOREIGN_KEY]
        assert len(tfk_edges) == 1

    def test_same_table_edges(self):
        graph = self._sample_graph()
        same = [e for e in graph.edges if e.edge_type == EdgeType.SAME_TABLE]
        # orders: C(3,2)=3 pairs, customers: C(2,2)=1 pair = 4 total
        assert len(same) == 4

    def test_neighbors(self):
        graph = self._sample_graph()
        neighbors = graph.neighbors("table:orders", EdgeType.COLUMN_OF)
        assert len(neighbors) >= 1  # At least non-PK columns

    def test_adjacency_list(self):
        graph = self._sample_graph()
        adj = graph.adjacency_list()
        assert len(adj) > 0

    def test_serialization(self):
        graph = self._sample_graph()
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 7

    def test_list_schema_format(self):
        graph = SchemaGraph.from_schema_dict(
            {"users": ["id", "name", "email"]},
        )
        assert graph.num_nodes == 4  # 1 table + 3 columns


# ── Identifier Decomposer ───────────────────────────────────────────

class TestIdentifierDecomposer:
    def test_snake_case(self):
        dec = IdentifierDecomposer()
        assert dec.decompose("customer_id") == ["customer", "id"]

    def test_camel_case(self):
        dec = IdentifierDecomposer()
        tokens = dec.decompose("customerId")
        assert "customer" in tokens or "Id" in [t.title() for t in tokens]

    def test_abbreviation_expansion(self):
        dec = IdentifierDecomposer()
        expanded = dec.expand_abbreviations("cust_id")
        assert "customer" in expanded
        assert "identifier" in expanded

    def test_single_word(self):
        dec = IdentifierDecomposer()
        assert dec.decompose("name") == ["name"]


# ── Grammar State Machine ───────────────────────────────────────────

class TestGrammarStateMachine:
    def test_valid_simple_sequence(self):
        fsm = GrammarStateMachine()
        sequence = [
            GrammarState.SELECT,
            GrammarState.FROM,
            GrammarState.WHERE,
            GrammarState.LIMIT,
            GrammarState.COMPLETE,
        ]
        valid, idx = fsm.validate_sequence(sequence)
        assert valid is True

    def test_valid_aggregate_sequence(self):
        fsm = GrammarStateMachine()
        sequence = [
            GrammarState.SELECT,
            GrammarState.FROM,
            GrammarState.GROUP_BY,
            GrammarState.HAVING,
            GrammarState.ORDER_BY,
            GrammarState.LIMIT,
            GrammarState.COMPLETE,
        ]
        valid, idx = fsm.validate_sequence(sequence)
        assert valid is True

    def test_invalid_having_before_group_by(self):
        fsm = GrammarStateMachine()
        sequence = [
            GrammarState.SELECT,
            GrammarState.FROM,
            GrammarState.HAVING,  # Invalid without GROUP_BY first
        ]
        valid, idx = fsm.validate_sequence(sequence)
        assert valid is False
        assert idx == 2

    def test_invalid_select_after_limit(self):
        fsm = GrammarStateMachine()
        sequence = [
            GrammarState.SELECT,
            GrammarState.FROM,
            GrammarState.LIMIT,
            GrammarState.SELECT,  # Invalid
        ]
        valid, idx = fsm.validate_sequence(sequence)
        assert valid is False

    def test_cte_before_select(self):
        fsm = GrammarStateMachine()
        sequence = [
            GrammarState.CTE,
            GrammarState.SELECT,
            GrammarState.FROM,
            GrammarState.COMPLETE,
        ]
        valid, idx = fsm.validate_sequence(sequence)
        assert valid is True

    def test_join_sequence(self):
        fsm = GrammarStateMachine()
        sequence = [
            GrammarState.SELECT,
            GrammarState.FROM,
            GrammarState.JOIN,
            GrammarState.JOIN,
            GrammarState.WHERE,
            GrammarState.COMPLETE,
        ]
        valid, idx = fsm.validate_sequence(sequence)
        assert valid is True

    def test_token_mask_changes_with_state(self):
        fsm = GrammarStateMachine()
        start_mask = fsm.token_mask()
        fsm.transition(GrammarState.SELECT)
        select_mask = fsm.token_mask()
        assert start_mask != select_mask
        assert TokenCategory.COLUMN_NAME in select_mask

    def test_reset(self):
        fsm = GrammarStateMachine()
        fsm.transition(GrammarState.SELECT)
        fsm.reset()
        assert fsm.state == GrammarState.START
        assert len(fsm.history) == 0

    def test_error_state_on_invalid_transition(self):
        fsm = GrammarStateMachine()
        result = fsm.transition(GrammarState.LIMIT)  # Invalid from START
        assert result is False
        assert fsm.is_error is True
