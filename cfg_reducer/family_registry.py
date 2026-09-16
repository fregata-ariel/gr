"""Explicit registration of deterministic CFG generator families."""

from contextlib import contextmanager
from collections.abc import Iterator
from importlib import import_module
from random import Random
from typing import Protocol

from .generator_types import CFGShape, GeneratorSpec, _validate_family


class CFGFamily(Protocol):
    @property
    def name(self) -> str: ...

    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec: ...

    def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape: ...


_families: dict[str, CFGFamily] = {}
_generation_depth = 0


def register_family(plugin: CFGFamily) -> None:
    if _generation_depth:
        raise ValueError("cannot register families during generation")
    _validate_family(plugin.name)
    if plugin.name in _families:
        raise ValueError(f"duplicate family: {plugin.name!r}")
    _families[plugin.name] = plugin


def get_family(name: str) -> CFGFamily:
    _validate_family(name)
    try:
        return _families[name]
    except KeyError:
        raise ValueError(f"unknown family: {name!r}") from None


def family_names() -> tuple[str, ...]:
    return tuple(sorted(_families))


def load_plugins(modules: tuple[str, ...]) -> None:
    """Import modules in order; each module explicitly registers its plugins."""
    if _generation_depth:
        raise ValueError("cannot load plugins during generation")
    for module in modules:
        import_module(module)


@contextmanager
def _generation_scope() -> Iterator[None]:
    global _generation_depth
    _generation_depth += 1
    try:
        yield
    finally:
        _generation_depth -= 1
