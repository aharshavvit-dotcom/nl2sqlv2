"""Deterministic decomposition of enterprise schema identifiers.

Converts identifiers like 'order_items', 'totalRevenue', 'first_name'
into meaningful token sequences for the vocabulary, improving
generalization across different naming conventions.

Examples:
    'order_items'  → ['order', 'items']
    'totalRevenue' → ['total', 'revenue']
    'first_name'   → ['first', 'name']
    'CUSTOMER_ID'  → ['customer', 'id']
    'orderDate'    → ['order', 'date']
    'is_active'    → ['is', 'active']
"""
from __future__ import annotations

import re
from functools import lru_cache


# Pattern for camelCase / PascalCase splitting
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Pattern for common separators
_SEPARATOR_SPLIT = re.compile(r"[_\-.\s/]+")

# Common abbreviations to keep together
_KNOWN_ABBREVIATIONS = frozenset({
    "id", "pk", "fk", "db", "sql", "ir", "api", "url", "ui",
    "qty", "amt", "num", "cnt", "avg", "max", "min", "sum",
    "dt", "ts", "yr", "mo", "dy",
})

# Stop words that add no semantic value for schema understanding
_STOP_TOKENS = frozenset({"the", "a", "an", "of", "in", "for", "to"})


@lru_cache(maxsize=4096)
def decompose_identifier(identifier: str) -> list[str]:
    """Decompose a schema identifier into meaningful tokens.

    Parameters
    ----------
    identifier : str
        A table or column name (e.g., 'order_items', 'totalRevenue').

    Returns
    -------
    list[str]
        Lowercased decomposed tokens.
    """
    if not identifier:
        return []

    # Step 1: Split on separators (underscore, dash, dot, space, slash)
    parts = _SEPARATOR_SPLIT.split(identifier)

    # Step 2: Further split each part on camelCase boundaries
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        sub_parts = _CAMEL_SPLIT.split(part)
        tokens.extend(sub_parts)

    # Step 3: Lowercase and filter
    result = []
    for token in tokens:
        token = token.lower().strip()
        if not token:
            continue
        if token in _STOP_TOKENS:
            continue
        result.append(token)

    return result


def decompose_and_flatten(identifier: str, max_tokens: int = 6) -> list[str]:
    """Decompose and truncate to max_tokens.

    Parameters
    ----------
    identifier : str
        Schema identifier to decompose.
    max_tokens : int
        Maximum number of tokens to return.

    Returns
    -------
    list[str]
        Truncated list of decomposed tokens.
    """
    tokens = decompose_identifier(identifier)
    return tokens[:max_tokens]


def decompose_schema_identifiers(
    tables: dict[str, dict],
    max_tokens_per_name: int = 4,
) -> dict[str, dict[str, list[str]]]:
    """Decompose all identifiers in a schema dict.

    Parameters
    ----------
    tables : dict
        Schema dictionary with table -> columns structure.
    max_tokens_per_name : int
        Max tokens per identifier.

    Returns
    -------
    dict mapping table_name -> {
        "table_tokens": [...],
        "columns": {col_name: [...], ...}
    }
    """
    result: dict[str, dict[str, list[str] | dict]] = {}
    for table_name, table_info in tables.items():
        table_tokens = decompose_and_flatten(table_name, max_tokens_per_name)
        columns_decomposed: dict[str, list[str]] = {}
        col_info = table_info if isinstance(table_info, dict) else {}
        raw_columns = col_info.get("columns") or {}
        for col_name in raw_columns:
            columns_decomposed[col_name] = decompose_and_flatten(
                col_name, max_tokens_per_name
            )
        result[table_name] = {
            "table_tokens": table_tokens,
            "columns": columns_decomposed,
        }
    return result
