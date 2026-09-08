import json

import pytest

from cfg_reducer.buckets import (
    AcceptDecision, BucketDimension, BucketPlan, CFGReference, bucket_acceptor,
)
from cfg_reducer.dataset import build_dataset
from cfg_reducer.generator_types import GenerationRejected


def toy(engine, *, seed):
    if seed == 5:
        raise GenerationRejected('node_budget')
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 6: 4}[seed]
    nodes = [f'n{i}' for i in range(size)]
    for node in nodes:
        engine.add_node(node)
    for a, b in zip(nodes, nodes[1:]):
        engine.add_edge(a, b)
    return nodes


def test_accept_order_and_counts(tmp_path):
    plan = BucketPlan((BucketDimension('num_nodes', (), ('all',)),), 3)
    inner = bucket_acceptor(plan)
    states = []

    def accept(candidate, state):
        states.append((candidate.seed, state.counts))
        result = inner(candidate, state)
        if candidate.seed == 2:
            return AcceptDecision(False, 'outside_plan', result.bucket, result.realized)
        return result

    setattr(accept, "plan", plan)
    result = build_dataset(tmp_path / 'a', {'train': (0, 7)}, {}, 'test', toy, accept=accept)
    split = result['splits']['train']
    assert [s['seed'] for s in split['samples']] == [0, 3, 4]
    assert states == [(0, ()), (1, (('num_nodes=all', 1),)), (2, (('num_nodes=all', 1),)),
                      (3, (('num_nodes=all', 1),)), (4, (('num_nodes=all', 2),)),
                      (6, (('num_nodes=all', 3),))]
    assert split['complete']
    assert split['per_bucket']['num_nodes=all'] == dict(target=3, attempts=6, accepted=3, rejected=3, missing=0)
    assert split['rejected_by_reason'] == dict(duplicate=1, outside_plan=1, node_budget=1, bucket_full=1)
    rows = [json.loads(line) for line in (tmp_path / 'a/rejections.jsonl').read_text().splitlines()]
    assert len(rows) == result['rejected'] == 4
    assert rows[0]['realized']['num_nodes'] == 1
    assert rows[2]['bucket'] is rows[2]['realized'] is None
    assert 'rejections' not in split
    again = build_dataset(tmp_path / 'b', {'train': (0, 7)}, {}, 'test', toy, accept=accept)
    assert again == result
    for path in (tmp_path / 'a').rglob('*.json*'):
        assert path.read_bytes() == (tmp_path / 'b' / path.relative_to(tmp_path / 'a')).read_bytes()

    def fail(engine, *, seed):
        raise GenerationRejected('node_budget')

    failed = build_dataset(tmp_path / 'fail', {'train': (0, 3)}, {}, 'test', fail, accept=inner)['splits']['train']
    assert not failed['complete']
    assert failed['attempts'] == failed['rejected'] == 3
    assert failed['per_bucket']['num_nodes=all']['missing'] == 3

    ref = CFGReference('other', 'original', ('renamed',), ())
    excluded = build_dataset(tmp_path / 'exclude', {'train': (0, 4)}, {}, 'test', toy, exclude=(ref,))
    assert [s['seed'] for s in excluded['splits']['train']['samples']] == [2]
    assert excluded['splits']['train']['rejected_by_reason'] == {'cross_dataset_duplicate': 2, 'duplicate': 1}
    assert excluded['selection']['excluded_datasets'] == ['other']
    assert excluded['selection']['plans'] == {'train': None}


def test_duplicate_priority_and_cross_split(tmp_path):
    hook = bucket_acceptor(BucketPlan((BucketDimension('num_nodes', (), ('all',)),), 1))
    result = build_dataset(tmp_path, {'train': (0, 1), 'test': (1, 3)}, {}, 'test', toy, accept=hook)
    assert result['splits']['test']['rejected_by_reason'] == {'duplicate': 1}
    assert result['splits']['test']['accepted'] == 1


def test_generator_bugs_propagate(tmp_path):
    def broken(engine, *, seed):
        raise TypeError('bug')
    with pytest.raises(TypeError, match='bug'):
        build_dataset(tmp_path, {'test': (0, 1)}, {}, 'test', broken,
                      accept=bucket_acceptor(BucketPlan((BucketDimension('num_nodes', (), ('all',)),), 1)))
