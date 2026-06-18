from __future__ import annotations

import re


def normalize_identifier(value: str) -> str:
    """Normalize an identifier for matching while keeping it identifier-shaped."""
    normalized = str(value or "").strip().strip('"`[]').lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def tokenize_identifier(value: str) -> list[str]:
    normalized = str(value or "").strip().strip('"`[]').lower()
    normalized = re.sub(r"[_\-]+", " ", normalized)
    return re.findall(r"[a-z0-9]+", normalized)


def singularize_simple(token: str) -> str:
    token = token.lower().strip()
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _pluralize_simple(token: str) -> str:
    token = token.lower().strip()
    if not token:
        return token
    if token.endswith("y") and len(token) > 3:
        return token[:-1] + "ies"
    if token.endswith("s"):
        return token
    return token + "s"


def normalize_table_phrase(value: str) -> str:
    tokens = tokenize_identifier(value)
    return " ".join(tokens)


def table_name_variants(table_name: str) -> set[str]:
    ident = normalize_identifier(table_name)
    tokens = tokenize_identifier(table_name)
    singular_tokens = [singularize_simple(token) for token in tokens]
    variants = {
        ident,
        ident.replace("_", " "),
        " ".join(tokens),
        " ".join(singular_tokens),
    }
    if tokens:
        variants.update(tokens)
        variants.update(singular_tokens)
        variants.update(_pluralize_simple(token) for token in singular_tokens)
        variants.add("_".join(singular_tokens))
    if len(tokens) > 1:
        variants.add(tokens[-1])
        variants.add(singular_tokens[-1])
        variants.add(tokens[0])
        variants.add(singular_tokens[0])
    return {variant.strip().lower() for variant in variants if variant and variant.strip()}


def column_name_variants(column_name: str) -> set[str]:
    ident = normalize_identifier(column_name)
    tokens = tokenize_identifier(column_name)
    singular_tokens = [singularize_simple(token) for token in tokens]
    variants = {
        ident,
        ident.replace("_", " "),
        " ".join(tokens),
        " ".join(singular_tokens),
    }
    if tokens:
        variants.update(tokens)
        variants.update(singular_tokens)
    return {variant.strip().lower() for variant in variants if variant and variant.strip()}
