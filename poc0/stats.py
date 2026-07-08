from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from cfg_reducer.types import Skeleton

_KIND_NAMES = ("entry", "linear", "merge", "loop")


def _count_kinds_recursive(skeleton: Skeleton) -> Counter[str]:
    counts = Counter(kind for kind, _ in skeleton.items)
    for subgraph in skeleton.subgraphs.values():
        counts.update(_count_kinds_recursive(subgraph))
    return counts


def _top_level_shape(skeleton: Skeleton) -> tuple[int, int]:
    if not skeleton.items:
        return 0, 0

    parents: dict[int, set[int]] = {}
    children: dict[int, list[int]] = {}
    for index, (_, parent_indices) in enumerate(skeleton.items):
        parent_set = set(parent_indices)
        parents[index] = parent_set
        children[index] = []

    for child_index, parent_set in parents.items():
        for parent_index in parent_set:
            if parent_index < 0 or parent_index >= len(skeleton.items):
                raise ValueError(
                    f"parent index {parent_index} is out of range for skeleton item {child_index}"
                )
            children[parent_index].append(child_index)

    indegree = {index: len(parent_set) for index, parent_set in parents.items()}
    current = [index for index, degree in indegree.items() if degree == 0]
    depth = 0
    max_width = 0
    visited = 0

    while current:
        depth += 1
        max_width = max(max_width, len(current))
        next_layer: list[int] = []
        for index in current:
            visited += 1
            for child_index in children[index]:
                indegree[child_index] -= 1
                if indegree[child_index] == 0:
                    next_layer.append(child_index)
        current = next_layer

    if visited != len(skeleton.items):
        raise ValueError("skeleton parent indices must form a DAG")

    return depth, max_width


def _loop_nest(skeleton: Skeleton) -> int:
    best = 0
    for subgraph in skeleton.subgraphs.values():
        best = max(best, 1 + _loop_nest(subgraph))
    return best


def stats_for_skeleton(
    skeleton: Skeleton,
    tokens: list[str],
) -> dict[str, int | dict[str, int]]:
    counts = _count_kinds_recursive(skeleton)
    depth, max_width = _top_level_shape(skeleton)
    return {
        "n_motifs": sum(counts.values()),
        "kinds": {kind: counts[kind] for kind in _KIND_NAMES if counts[kind] > 0},
        "n_tokens": len(tokens),
        "depth": depth,
        "max_width": max_width,
        "loop_nest": _loop_nest(skeleton),
    }


def normalized_histogram(values: Iterable[int]) -> dict[int, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        key: value / total
        for key, value in sorted(counts.items())
    }


def total_variation_distance(
    p: Mapping[int, float],
    q: Mapping[int, float],
) -> float:
    bins = set(p) | set(q)
    return 0.5 * sum(abs(p.get(bin_value, 0.0) - q.get(bin_value, 0.0)) for bin_value in bins)
