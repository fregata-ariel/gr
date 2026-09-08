"""Measure realized CFG and MetaGraph features for dataset candidates."""

from dataclasses import dataclass

from .generator_types import Features, Json
from .reducibility import is_reducible
from .structure_features import features_for
from .types import MetaGraph


@dataclass(frozen=True)
class Candidate:
    split: str
    seed: int
    sample_id: str
    requested: dict[str, Json]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    mg: MetaGraph


def measure_realized(candidate: Candidate) -> Features:
    return features_for(candidate.mg) | {
        "num_nodes": len(candidate.nodes),
        "reducible": is_reducible(candidate.nodes, candidate.edges, entry=candidate.nodes[0]),
    }
