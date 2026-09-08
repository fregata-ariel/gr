"""Layered DAGs with loop and goto attempts, preserving the v1 RNG sequence."""

from random import Random
from typing import cast

from ..generator_types import CFGShape, GeneratorSpec, Json


_DEFAULTS: dict[str, Json] = {
    "max_layer_width": 3,
    "edge_prob": 0.5,
    "loop_count": 2,
    "goto_count": 1,
    "span_mode": "uniform",
}


def _choose_target(rng: Random, lo: int, hi: int, mode: str) -> int:
    if mode == "uniform":
        return rng.randint(lo, hi)
    count = max(1, (hi - lo + 3) // 3)
    candidates = (list(range(lo, lo + count)) if mode == "short"
                  else list(range(hi - count + 1, hi + 1)))
    return rng.choice(candidates)


class Layered:
    name = "layered"

    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec:
        # Revalidate even if a caller has mutated the spec's params dictionary.
        spec = GeneratorSpec(spec.family, spec.num_nodes, spec.params)
        if spec.family != self.name:
            raise ValueError("layered requires family='layered'")
        unknown = set(spec.params) - _DEFAULTS.keys()
        if unknown:
            raise ValueError(f"unknown layered params: {sorted(unknown)}")
        params = _DEFAULTS | spec.params
        for key, minimum in (("max_layer_width", 1), ("loop_count", 0),
                             ("goto_count", 0)):
            value = params[key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        probability = params["edge_prob"]
        if (isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1):
            raise ValueError("edge_prob must be a number in [0, 1]")
        if params["span_mode"] not in ("uniform", "short", "long"):
            raise ValueError("span_mode must be uniform, short, or long")
        return GeneratorSpec(self.name, spec.num_nodes, params)

    def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
        spec = self.normalize(spec)
        n = spec.num_nodes
        ids = tuple(f"N{i:02d}" for i in range(n))
        if n < 3:
            return CFGShape(ids, tuple(zip(ids, ids[1:])), ids[0])

        width = cast(int, spec.params["max_layer_width"])
        probability = cast(int | float, spec.params["edge_prob"])
        weights = [size for size in range(1, width) for _ in range(2)] + [width]
        edges: set[tuple[str, str]] = set()
        current = ids[:1]
        remaining = ids[1:-1]
        while remaining:
            size = min(len(remaining), rng.choice(weights))
            next_layer, remaining = remaining[:size], remaining[size:]
            for u in current:
                edges.add((u, rng.choice(next_layer)))
            for v in next_layer:
                if not any((u, v) in edges for u in current):
                    edges.add((rng.choice(current), v))
            for u in current:
                for v in next_layer:
                    # Consume a draw even when the edge already exists.
                    if rng.random() < probability and (u, v) not in edges:
                        edges.add((u, v))
            current = next_layer
        for node in current:
            edges.add((node, ids[-1]))

        # Counts are attempts: duplicate edges never trigger another draw.
        for _ in range(cast(int, spec.params["loop_count"])):
            j = rng.randint(2, n - 2)
            i = rng.randint(0, j - 2)
            edges.add((ids[j], ids[i]))
        for _ in range(cast(int, spec.params["goto_count"])):
            i = rng.randint(1, n - 3)
            j = _choose_target(rng, i + 2, n - 1,
                               cast(str, spec.params["span_mode"]))
            edges.add((ids[i], ids[j]))
        return CFGShape(ids, tuple(sorted(edges)), ids[0])
