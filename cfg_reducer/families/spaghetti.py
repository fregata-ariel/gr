"""Structured CFGs with uniformly sampled additional edges."""

import math
from random import Random
from typing import cast

from ..generator_types import CFGShape, GeneratorSpec
from .structured import Structured


class Spaghetti:
    name = "spaghetti"

    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec:
        spec = GeneratorSpec(spec.family, spec.num_nodes, spec.params)
        if spec.family != self.name:
            raise ValueError("spaghetti requires family='spaghetti'")
        params = dict(spec.params)
        rate = params.pop("spaghetti_rate", 0.1)
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not 0 <= rate <= 1):
            raise ValueError("spaghetti_rate must be a finite number in [0, 1]")
        base = Structured().normalize(GeneratorSpec("structured", spec.num_nodes, params))
        return GeneratorSpec(self.name, spec.num_nodes,
                             base.params | {"spaghetti_rate": rate})

    def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
        spec = self.normalize(spec)
        params = dict(spec.params)
        rate = cast(int | float, params.pop("spaghetti_rate"))
        base = Structured().generate(GeneratorSpec("structured", spec.num_nodes, params), rng)
        edges = set(base.edges)
        budget = math.floor(rate * len(base.edges))
        candidates = sorted((u, v) for u in base.nodes for v in base.nodes
                            if u != v and (u, v) not in edges
                            and v != base.entry and u != base.nodes[-1])
        extra = rng.sample(candidates, min(budget, len(candidates)))
        return CFGShape(base.nodes, tuple(sorted(edges.union(extra))), base.entry)
