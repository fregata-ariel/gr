"""Data contracts shared by CFG generator plugins."""

from dataclasses import dataclass, field
import math
import re


type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Features = dict[str, int | float | bool]


def _copy_json(value: object) -> Json:
    """Validate and copy JSON values, producing recursively sorted keys."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _copy_json(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _copy_params(value: object) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError("params must be an object")
    result = _copy_json(value)
    assert isinstance(result, dict)
    return result


def _validate_family(name: object) -> None:
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("family must match ^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class GeneratorSpec:
    family: str
    num_nodes: int
    params: dict[str, Json] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_family(self.family)
        if type(self.num_nodes) is not int or self.num_nodes < 1:
            raise ValueError("num_nodes must be a positive integer")
        object.__setattr__(self, "params", _copy_params(self.params))


@dataclass(frozen=True)
class CFGShape:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    entry: str
