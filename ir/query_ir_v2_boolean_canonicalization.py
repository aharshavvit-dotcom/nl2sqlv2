from __future__ import annotations

from typing import Any

from .query_ir_v2_models import BooleanPredicate, NotPredicate, Predicate


def canonicalize_predicate(predicate: Predicate) -> Predicate:
    if isinstance(predicate, BooleanPredicate):
        operands = [canonicalize_predicate(item) for item in predicate.operands]
        flattened: list[Any] = []
        for operand in operands:
            if isinstance(operand, BooleanPredicate) and operand.operator == predicate.operator:
                flattened.extend(operand.operands)
            else:
                flattened.append(operand)
        if len(flattened) == 1:
            return flattened[0]
        return BooleanPredicate(
            operator=predicate.operator,
            operands=flattened,
            legacy_v1=dict(predicate.legacy_v1 or {}),
        )
    if isinstance(predicate, NotPredicate):
        return NotPredicate(
            operand=canonicalize_predicate(predicate.operand),
            legacy_v1=dict(predicate.legacy_v1 or {}),
        )
    return predicate


__all__ = ["canonicalize_predicate"]
