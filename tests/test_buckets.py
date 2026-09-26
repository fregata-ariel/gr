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


from cfg_reducer import buckets
from cfg_reducer.buckets import (
    AcceptanceState, BucketDimension, BucketPlan, bucket_for, bucket_acceptor,
    default_bucket_plan, ranges_acceptor, measurement_acceptor,
)
from cfg_reducer.types import MetaGraph
import math


def test_bucket_boundaries():
    plan = default_bucket_plan(1)
    features = {'mean_offset': 1.0, 'max_depth': 0, 'max_in_degree': 0, 'reducible': False}
    for dim in plan.dims[:-1]:
        for i, cut in enumerate(dim.cuts):
            for value, label in ((math.nextafter(cut, -math.inf), dim.labels[i]),
                                 (cut, dim.labels[i]),
                                 (math.nextafter(cut, math.inf), dim.labels[i + 1])):
                assert f'{dim.feature}={label}' in bucket_for(features | {dim.feature: value}, plan).split('|')
    assert bucket_for(features | {'reducible': True}, plan).endswith('reducible=yes')


def test_bucket_plan_determinism(monkeypatch):
    plan = BucketPlan((BucketDimension('max_depth', (1,), ('low', 'high')),), 1, (('low',),))
    hook = bucket_acceptor(plan)
    candidate = Candidate('train', 0, 'id', {}, ('a',), (), MetaGraph((), (), {}))
    state = AcceptanceState((('max_depth=low', 1),))
    for features, reason in (({'max_depth': 1}, 'bucket_full'),
                             ({'max_depth': 2}, 'outside_plan'),
                             ({}, 'invalid_feature'),
                             ({'max_depth': math.inf}, 'invalid_feature')):
        monkeypatch.setattr(buckets, 'measure_realized', lambda c: features)
        a = hook(candidate, state)
        assert a == hook(candidate, state)
        assert a.reason == reason
        assert state.counts == (('max_depth=low', 1),)
    monkeypatch.setattr(buckets, 'measure_realized', lambda c: {'max_depth': 1})
    assert hook(candidate, AcceptanceState(())).accepted
    assert hook(candidate, AcceptanceState(())).accepted
    with pytest.raises(ValueError):
        default_bucket_plan(0)


@pytest.mark.parametrize('value,accepted', [(0, False), (1, True), (2, True), (3, False)])
def test_ranges_inclusive(monkeypatch, value, accepted):
    monkeypatch.setattr(buckets, 'measure_realized', lambda c: {'max_depth': value})
    candidate = Candidate('train', 0, 'id', {}, ('a',), (), MetaGraph((), (), {}))
    hook = ranges_acceptor(measurement_acceptor, {'max_depth': (1, 2)})
    assert hook(candidate, AcceptanceState(())).accepted == accepted
    assert ranges_acceptor(measurement_acceptor, {'max_depth': (None, None)})(candidate, AcceptanceState(())).accepted


@pytest.mark.parametrize('factory', [
    lambda: BucketDimension('unknown', (), ('all',)),
    lambda: BucketDimension('max_depth', (1, 1), ('a', 'b', 'c')),
    lambda: BucketDimension('max_depth', (math.inf,), ('a', 'b')),
    lambda: BucketDimension('max_depth', (1,), ('a', 'a')),
    lambda: BucketDimension('max_depth', (), ('a=b',)),
    lambda: BucketDimension('max_depth', (), ('a', 'b')),
    lambda: BucketPlan((BucketDimension('max_depth', (), ('a',)),) * 2, 1),
    lambda: BucketPlan((BucketDimension('max_depth', (), ('a',)),), 1, ()),
    lambda: BucketPlan((BucketDimension('max_depth', (), ('a',)),), 1, (('b',),)),
    lambda: BucketPlan((BucketDimension('max_depth', (), ('a',)),), 1, (('a',), ('a',))),
])
def test_invalid_plans(factory):
    with pytest.raises(ValueError):
        factory()
