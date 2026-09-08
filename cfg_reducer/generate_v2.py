"""Validated spec codecs and family dispatch for CFG generation v2."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from . import families as _families  # Execute explicit built-in registrations.
from .engine import GraphEngine
from .family_registry import _generation_scope, get_family
from .generator_types import CFGShape, GeneratorSpec, Json, _copy_params

if TYPE_CHECKING:
    from .dataset import GeneratorDescriptor


def normalize_spec(spec: GeneratorSpec) -> GeneratorSpec:
    validated = GeneratorSpec(spec.family, spec.num_nodes, spec.params)
    normalized = get_family(validated.family).normalize(validated)
    if not isinstance(normalized, GeneratorSpec) or normalized.family != spec.family:
        raise ValueError("normalize must preserve the spec family")
    return GeneratorSpec(normalized.family, normalized.num_nodes, normalized.params)


def spec_to_json(spec: GeneratorSpec) -> dict[str, Json]:
    validated = GeneratorSpec(spec.family, spec.num_nodes, spec.params)
    return {"family": validated.family, "num_nodes": validated.num_nodes,
            "params": _copy_params(validated.params)}


def spec_from_json(value: dict[str, Json]) -> GeneratorSpec:
    if not isinstance(value, dict) or set(value) != {"family", "num_nodes", "params"}:
        raise ValueError("spec requires exactly family, num_nodes, params")
    family, num_nodes = value["family"], value["num_nodes"]
    if not isinstance(family, str):
        raise ValueError("family must be a string")
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
        raise ValueError("num_nodes must be an integer")
    return normalize_spec(GeneratorSpec(family, num_nodes, _copy_params(value["params"])))


def _validate_shape(shape: CFGShape) -> None:
    if not all(isinstance(node, str) for node in shape.nodes):
        raise ValueError("node IDs must be strings")
    nodes = set(shape.nodes)
    if len(nodes) != len(shape.nodes):
        raise ValueError("duplicate node IDs")
    if shape.entry not in nodes:
        raise ValueError("entry must exist in nodes")
    if len(set(shape.edges)) != len(shape.edges):
        raise ValueError("duplicate edges")
    for src, dst in shape.edges:
        if src not in nodes or dst not in nodes:
            raise ValueError("unknown edge endpoint")


def generate_cfg_v2(engine: GraphEngine, *, seed: int, spec: GeneratorSpec) -> list[str]:
    if engine.nodes:
        raise ValueError("generation requires an empty engine")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    with _generation_scope():
        normalized = normalize_spec(spec)
        shape = get_family(normalized.family).generate(normalized, Random(seed))
        _validate_shape(shape)
    for node in shape.nodes:
        engine.add_node(node)
    for src, dst in sorted(shape.edges):
        engine.add_edge(src, dst)
    return list(shape.nodes)


def descriptor_for(spec: GeneratorSpec) -> GeneratorDescriptor:
    """Use with config={"spec": spec_to_json(normalize_spec(spec))}."""
    from .dataset import GeneratorDescriptor

    family = normalize_spec(spec).family

    def adapter(engine: GraphEngine, *, seed: int, spec: dict[str, Json]) -> list[str]:
        restored = spec_from_json(spec)
        if restored.family != family:
            raise ValueError("descriptor and spec family must match")
        return generate_cfg_v2(engine, seed=seed, spec=restored)

    return GeneratorDescriptor("cfg_v2:" + family, adapter)
