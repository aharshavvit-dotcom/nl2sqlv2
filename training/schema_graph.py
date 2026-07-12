"""Schema Graph with Structured Embeddings — Gate 3 Architecture.

Builds a typed graph representation of database schemas for GNN-based
encoding. Nodes are tables and columns; edges encode PK-FK, column-table
membership, and type relationships.

All behind `enable_schema_graph` feature flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    TABLE = "table"
    COLUMN = "column"


class EdgeType(str, Enum):
    COLUMN_OF = "column_of"          # column -> table
    PRIMARY_KEY = "primary_key"       # column -> table (PK)
    FOREIGN_KEY = "foreign_key"       # column -> column
    SAME_TABLE = "same_table"         # column -> column (same table)
    TABLE_FOREIGN_KEY = "table_fk"    # table -> table (via FK)


@dataclass
class SchemaNode:
    """A node in the schema graph (table or column)."""
    node_id: str
    node_type: NodeType
    name: str
    table_name: str | None = None  # For column nodes
    data_type: str = ""
    is_primary_key: bool = False
    is_nullable: bool = True
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        if self.table_name:
            return f"{self.table_name}.{self.name}"
        return self.name


@dataclass
class SchemaEdge:
    """An edge in the schema graph."""
    source: str  # node_id
    target: str  # node_id
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SchemaGraph:
    """Typed graph representation of a database schema.

    Usage:
        graph = SchemaGraph.from_schema_dict(schema)
        nodes = graph.get_column_nodes("orders")
        adj = graph.adjacency_list()
    """

    def __init__(self) -> None:
        self._nodes: dict[str, SchemaNode] = {}
        self._edges: list[SchemaEdge] = []
        self._adjacency: dict[str, list[tuple[str, EdgeType]]] = {}

    def add_node(self, node: SchemaNode) -> None:
        self._nodes[node.node_id] = node

    def add_edge(self, edge: SchemaEdge) -> None:
        self._edges.append(edge)
        self._adjacency.setdefault(edge.source, []).append((edge.target, edge.edge_type))
        self._adjacency.setdefault(edge.target, []).append((edge.source, edge.edge_type))

    def get_node(self, node_id: str) -> SchemaNode | None:
        return self._nodes.get(node_id)

    def get_table_nodes(self) -> list[SchemaNode]:
        return [n for n in self._nodes.values() if n.node_type == NodeType.TABLE]

    def get_column_nodes(self, table_name: str | None = None) -> list[SchemaNode]:
        nodes = [n for n in self._nodes.values() if n.node_type == NodeType.COLUMN]
        if table_name:
            nodes = [n for n in nodes if n.table_name == table_name]
        return nodes

    def neighbors(self, node_id: str, edge_type: EdgeType | None = None) -> list[SchemaNode]:
        raw = self._adjacency.get(node_id, [])
        if edge_type:
            raw = [(nid, et) for nid, et in raw if et == edge_type]
        return [self._nodes[nid] for nid, _ in raw if nid in self._nodes]

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def edges(self) -> list[SchemaEdge]:
        return list(self._edges)

    def adjacency_list(self) -> dict[str, list[tuple[str, str]]]:
        return {
            src: [(tgt, et.value) for tgt, et in neighbors]
            for src, neighbors in self._adjacency.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type.value,
                    "name": n.name,
                    "table_name": n.table_name,
                    "data_type": n.data_type,
                    "is_primary_key": n.is_primary_key,
                    "features": n.features,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in self._edges
            ],
        }

    @classmethod
    def from_schema_dict(
        cls,
        schema: dict[str, Any],
        *,
        primary_keys: dict[str, str] | None = None,
        foreign_keys: list[tuple[str, str, str, str]] | None = None,
    ) -> SchemaGraph:
        """Build a schema graph from a schema dictionary.

        Args:
            schema: {table_name: {column_name: {"type": ...}}} or {table_name: [col_names]}
            primary_keys: {table_name: pk_column}
            foreign_keys: [(src_table, src_col, tgt_table, tgt_col)]
        """
        graph = cls()
        primary_keys = primary_keys or {}
        foreign_keys = foreign_keys or []

        # Add table and column nodes
        for table_name, columns in schema.items():
            table_id = f"table:{table_name}"
            graph.add_node(SchemaNode(
                node_id=table_id,
                node_type=NodeType.TABLE,
                name=table_name,
            ))

            col_names: list[str]
            if isinstance(columns, dict):
                col_names = list(columns.keys())
            elif isinstance(columns, (list, set)):
                col_names = list(columns)
            else:
                continue

            for col_name in col_names:
                col_id = f"col:{table_name}.{col_name}"
                is_pk = primary_keys.get(table_name) == col_name
                data_type = ""
                if isinstance(columns, dict) and isinstance(columns.get(col_name), dict):
                    data_type = columns[col_name].get("type", "")

                graph.add_node(SchemaNode(
                    node_id=col_id,
                    node_type=NodeType.COLUMN,
                    name=col_name,
                    table_name=table_name,
                    data_type=data_type,
                    is_primary_key=is_pk,
                ))

                # column_of edge
                edge_type = EdgeType.PRIMARY_KEY if is_pk else EdgeType.COLUMN_OF
                graph.add_edge(SchemaEdge(
                    source=col_id,
                    target=table_id,
                    edge_type=edge_type,
                ))

            # same_table edges between columns
            col_ids = [f"col:{table_name}.{c}" for c in col_names]
            for i, cid1 in enumerate(col_ids):
                for cid2 in col_ids[i + 1:]:
                    graph.add_edge(SchemaEdge(
                        source=cid1,
                        target=cid2,
                        edge_type=EdgeType.SAME_TABLE,
                    ))

        # Foreign key edges
        for src_table, src_col, tgt_table, tgt_col in foreign_keys:
            src_col_id = f"col:{src_table}.{src_col}"
            tgt_col_id = f"col:{tgt_table}.{tgt_col}"
            src_table_id = f"table:{src_table}"
            tgt_table_id = f"table:{tgt_table}"

            if src_col_id in graph._nodes and tgt_col_id in graph._nodes:
                graph.add_edge(SchemaEdge(
                    source=src_col_id,
                    target=tgt_col_id,
                    edge_type=EdgeType.FOREIGN_KEY,
                ))
            if src_table_id in graph._nodes and tgt_table_id in graph._nodes:
                graph.add_edge(SchemaEdge(
                    source=src_table_id,
                    target=tgt_table_id,
                    edge_type=EdgeType.TABLE_FOREIGN_KEY,
                ))

        return graph


# ── Identifier Decomposer ────────────────────────────────────────────

class IdentifierDecomposer:
    """Decomposes SQL identifiers into meaningful sub-tokens.

    Handles camelCase, snake_case, abbreviations, and numeric suffixes.
    This provides richer features for schema encoding than treating
    identifiers as opaque strings.
    """

    ABBREVIATION_MAP = {
        "id": "identifier",
        "pk": "primary key",
        "fk": "foreign key",
        "dt": "date",
        "ts": "timestamp",
        "qty": "quantity",
        "amt": "amount",
        "desc": "description",
        "num": "number",
        "cnt": "count",
        "avg": "average",
        "max": "maximum",
        "min": "minimum",
        "idx": "index",
        "ref": "reference",
        "grp": "group",
        "cat": "category",
        "dept": "department",
        "emp": "employee",
        "cust": "customer",
        "prod": "product",
        "inv": "inventory",
        "txn": "transaction",
        "addr": "address",
        "pmt": "payment",
    }

    def decompose(self, identifier: str) -> list[str]:
        """Split an identifier into meaningful sub-tokens."""
        # Split on underscores first
        parts = identifier.split("_")
        tokens: list[str] = []
        for part in parts:
            # Split camelCase
            tokens.extend(self._split_camel(part))

        # Lowercase and filter empties
        tokens = [t.lower() for t in tokens if t]
        return tokens

    def expand_abbreviations(self, identifier: str) -> str:
        """Decompose and expand known abbreviations."""
        tokens = self.decompose(identifier)
        expanded = [self.ABBREVIATION_MAP.get(t, t) for t in tokens]
        return " ".join(expanded)

    @staticmethod
    def _split_camel(text: str) -> list[str]:
        """Split camelCase or PascalCase into parts."""
        if not text:
            return []
        import re
        return re.sub(r'([A-Z])', r' \1', text).split()


__all__ = [
    "EdgeType",
    "IdentifierDecomposer",
    "NodeType",
    "SchemaEdge",
    "SchemaGraph",
    "SchemaNode",
]
