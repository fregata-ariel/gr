"""Weighted mixtures of registered CFG families with independent child RNGs."""

import math
from random import Random
from typing import cast

from ..family_registry import get_family
from ..generator_types import CFGShape, GeneratorSpec, Json


def _select(spec: GeneratorSpec, rng: Random) -> tuple[int, GeneratorSpec, Random]:
    components = cast(list[dict[str, Json]], spec.params["components"])
    weights = [cast(float, component["weight"]) for component in components]
    scale = max(weights)
    weights = [weight / scale for weight in weights]
    u = rng.random()
    threshold = u * math.fsum(weights)
    cumulative = 0.0
    index = len(components) - 1
    for i, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            index = i
            break
    component = components[index]
    child = GeneratorSpec(cast(str, component["family"]), spec.num_nodes,
                          cast(dict[str, Json], component["params"]))
    return index, child, Random(rng.getrandbits(64))


def component_for(spec: GeneratorSpec, seed: int) -> int:
    """Return the selected component index without generating its CFG."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return _select(Mixture().normalize(spec), Random(seed))[0]


class Mixture:
    name = "mixture"

    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec:
        spec = GeneratorSpec(spec.family, spec.num_nodes, spec.params)
        if spec.family != self.name:
            raise ValueError("mixture requires family='mixture'")
        if set(spec.params) != {"components"}:
            raise ValueError("mixture requires exactly components")
        components = spec.params["components"]
        if not isinstance(components, list) or not components:
            raise ValueError("components must be a nonempty list")
        normalized: list[Json] = []
        for component in components:
            if (not isinstance(component, dict)
                    or not {"family", "weight"} <= component.keys()
                    or component.keys() - {"family", "weight", "params"}):
                raise ValueError("components require family, weight, and optional params")
            family, weight = component["family"], component["weight"]
            params = component.get("params", {})
            if not isinstance(family, str) or family == self.name:
                raise ValueError("component family must be a string other than mixture")
            if not isinstance(params, dict):
                raise ValueError("component params must be an object")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError("weight must be a finite number > 0")
            try:
                weight = float(weight)
            except OverflowError:
                raise ValueError("weight must be a finite number > 0") from None
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("weight must be a finite number > 0")
            child = get_family(family).normalize(GeneratorSpec(family, spec.num_nodes, params))
            normalized.append({"family": family, "weight": weight, "params": child.params})
        return GeneratorSpec(self.name, spec.num_nodes, {"components": normalized})

    def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
        _, child, sub = _select(self.normalize(spec), rng)
        return get_family(child.family).generate(child, sub)
