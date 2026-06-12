from __future__ import annotations

import warnings
from typing import Any

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - fallback for pre-install test runs
    class _FieldInfo:
        def __init__(self, default: Any = None, default_factory: Any = None):
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = None, default_factory: Any = None) -> Any:
        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:
        def __init__(self, **kwargs: Any):
            annotations: dict[str, Any] = {}
            for cls in reversed(type(self).mro()):
                annotations.update(getattr(cls, "__annotations__", {}))
            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    default = getattr(type(self), name, None)
                    if isinstance(default, _FieldInfo):
                        if default.default_factory is not None:
                            value = default.default_factory()
                        else:
                            value = default.default
                    elif default is not None:
                        value = default
                    else:
                        raise TypeError(f"Missing required field: {name}")
                setattr(self, name, value)

        def model_dump(self) -> dict[str, Any]:
            return {
                name: _dump_value(getattr(self, name))
                for name in getattr(type(self), "__annotations__", {})
            }

        def dict(self) -> dict[str, Any]:
            return self.model_dump()


def _dump_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump_value(item) for key, item in value.items()}
    return value


warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "Text2SQLExample" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)


class DatabaseSchema(BaseModel):
    db_id: str
    dataset_name: str
    db_path: str | None = None
    tables: dict[str, Any] = Field(default_factory=dict)
    foreign_keys: list[Any] = Field(default_factory=list)
    primary_keys: list[Any] = Field(default_factory=list)
    serialized_schema: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class Text2SQLExample(BaseModel):
    example_id: str
    dataset_name: str
    db_id: str
    question: str
    sql: str
    split: str
    db_path: str | None = None
    schema: dict[str, Any] | None = None
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    template_id: str | None = None
    intent: str | None = None
    extracted_slots: dict[str, Any] = Field(default_factory=dict)
    sql_features: dict[str, Any] = Field(default_factory=dict)
    is_supported: bool = False
    unsupported_reason: str | None = None
    difficulty: str | None = None
    source_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SQLFeatures(BaseModel):
    statement_type: str | None = None
    selected_columns: list[str] = Field(default_factory=list)
    aggregations: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    joins: list[dict[str, Any]] = Field(default_factory=list)
    where_conditions: list[dict[str, Any]] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: list[dict[str, Any]] = Field(default_factory=list)
    limit: int | None = None
    has_nested_query: bool = False
    has_set_operation: bool = False
    has_having: bool = False
    has_window_function: bool = False
    complexity: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TrainingCorpusStats(BaseModel):
    total_examples: int = 0
    supported_examples: int = 0
    unsupported_examples: int = 0
    by_dataset: dict[str, int] = Field(default_factory=dict)
    by_template: dict[str, int] = Field(default_factory=dict)
    by_split: dict[str, int] = Field(default_factory=dict)
    unsupported_reasons: dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
