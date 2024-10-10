from collections.abc import Callable, Collection
from typing import Any, Literal, NamedTuple, Protocol


class SklearnEstimator(Protocol):
    def fit(self, X: Any, y: Any) -> Any:
        pass


class ColumnTransformerUnit(NamedTuple):
    name: str
    transformer: Literal["drop", "passthrough"] | SklearnEstimator
    columns: str | Collection[str] | int | Collection[int] | Collection[bool] | Callable
