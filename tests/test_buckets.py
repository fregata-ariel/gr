"""Realized measurements combine MetaGraph features with the original CFG."""

import pytest

from cfg_reducer import GraphEngine
from cfg_reducer.buckets import Candidate, measure_realized
from cfg_reducer.dataset import cfg_edges, cfg_nodes, reduce_to_metagraph
from cfg_reducer.generate import generate_cfg
from cfg_reducer.generate_v2 import generate_cfg_v2
from cfg_reducer.generator_types import GeneratorSpec
from cfg_reducer.structure_features import features_for, _walk_levels
from training import structure_features


@pytest.mark.parametrize('kind,expected_reducible', [
    ('v1', True), ('structured', True), ('multi_entry', False),
])
def test_realized_features(kind: str, expected_reducible: bool) -> None:
    engine = GraphEngine()
    if kind == 'v1':
        generate_cfg(engine, 2, seed=0)
    elif kind == 'structured':
        generate_cfg_v2(engine, seed=0, spec=GeneratorSpec('structured', 12))
    else:
        for node in ('s', 'a', 'b'):
            engine.add_node(node)
        for u, v in (('s', 'a'), ('s', 'b'), ('a', 'b'), ('b', 'a')):
            engine.add_edge(u, v)
    # Snapshot before reduction; entry is the first node, not lexical minimum.
    nodes = ('s', 'a', 'b') if kind == 'multi_entry' else tuple(cfg_nodes(engine))
    edges = tuple(cfg_edges(engine))
    mg = reduce_to_metagraph(engine)
    candidate = Candidate('test', 0, 'sample', {}, nodes, edges, mg)
    realized = measure_realized(candidate)
    original = features_for(mg)
    assert len(original) == 23
    assert len(realized) == 25
    assert realized == original | {'num_nodes': len(nodes), 'reducible': expected_reducible}
    assert realized['reducible'] is expected_reducible
    assert structure_features.features_for is features_for
    assert structure_features._walk_levels is _walk_levels
