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


def _legacy_fixture(out):
    """Regenerate the checked-in manifest through the unchanged v1 path."""
    return build_dataset(out, {'train': (0, 4), 'val': (4, 6)},
                         {'num_nodes': 8, 'edge_prob': 0.3}, version='fixture')


def test_manifest_legacy_bytes(tmp_path):
    from pathlib import Path
    from cfg_reducer.store import load_sample

    manifest = _legacy_fixture(tmp_path)
    fixture = Path(__file__).parent / 'fixtures/generator_v1_manifest.json'
    assert (tmp_path / 'manifest.json').read_bytes() == fixture.read_bytes()
    assert json.dumps(manifest, indent=2, ensure_ascii=False).encode() == fixture.read_bytes()
    expected = json.loads(fixture.read_bytes())
    for split, info in expected['splits'].items():
        payloads = [json.loads(p.read_bytes()) for p in sorted((tmp_path / split).glob('*.json'))]
        assert sorted(p['sample_id'] for p in payloads) == sorted(s['sample_id'] for s in info['samples'])
        for path in (tmp_path / split).glob('*.json'):
            load_sample(path)


def test_v2_cli_and_cross_dataset_dedup(tmp_path, monkeypatch):
    import hashlib
    import sys
    from cfg_reducer import family_registry
    from cfg_reducer.dataset_v2 import main, load_references

    monkeypatch.setattr(family_registry, '_families', dict(family_registry._families))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'family': 'layered', 'num_nodes': 8, 'params': {'edge_prob': 1}}))
    plan_path = tmp_path / 'plans.json'
    plan = {'dims': [{'feature': 'num_nodes', 'cuts': [8], 'labels': ['small', 'large']}],
            'target_per_bucket': 1, 'active': [['small']], 'ranges': {'num_nodes': [8, 8]}}

    def run(name, *extra, span='train=25:26'):
        out = tmp_path / name
        main(['--spec', str(spec), '--out', str(out), '--split', span,
              '--version', 'test', *extra])
        return json.loads((out / 'manifest.json').read_bytes())

    a = run('a')
    a_id = hashlib.sha256((tmp_path / 'a/manifest.json').read_bytes()).hexdigest()
    b = run('b', '--exclude-dataset', str(tmp_path / 'a'), span='train=68:69')
    assert b['selection']['excluded_datasets'] == [a_id]
    assert b['splits']['train']['rejected_by_reason'] == {'cross_dataset_duplicate': 1}
    rows = [json.loads(line) for line in (tmp_path / 'b/rejections.jsonl').read_text().splitlines()]
    assert all(row['duplicate_dataset'] == a_id and row['reason'] == 'cross_dataset_duplicate' for row in rows)
    assert all(row['duplicate_of'] == a['splits']['train']['samples'][0]['sample_id'] for row in rows)
    assert 'rejections' not in b['splits']['train']
    plan_path.write_text(json.dumps({'train': plan, 'val': None}))
    planned = run('planned', '--plan', str(plan_path), '--split', 'val=3:4', '--split', 'test=5:6')
    assert planned['selection']['plans'] == {'train': plan, 'val': None, 'test': None}
    assert planned['splits']['train']['complete']
    assert planned['splits']['train']['accepted'] == 1
    plan['ranges'] = {'num_nodes': [None, 7]}
    plan_path.write_text(json.dumps({'train': plan}))
    with pytest.raises(SystemExit) as exc:
        run('incomplete', '--plan', str(plan_path))
    assert exc.value.code == 2
    incomplete = json.loads((tmp_path / 'incomplete/manifest.json').read_bytes())
    assert incomplete['splits']['train']['rejected_by_reason'] == {'outside_plan': 1}
    assert not incomplete['splits']['train']['complete']
    assert run('allowed', '--plan', str(plan_path), '--allow-incomplete')['accepted'] == 0
    empty = run('empty_reference', '--exclude-dataset', str(tmp_path / 'incomplete'))
    assert empty['selection']['excluded_datasets'] == [hashlib.sha256(
        (tmp_path / 'incomplete/manifest.json').read_bytes()).hexdigest()]
    plan_path.write_text(json.dumps({'train': {'ranges': {'num_nodes': [8, None]}}}))
    assert run('ranges_only', '--plan', str(plan_path))['accepted'] == 1
    with pytest.raises(ValueError, match='version'):
        load_references((tmp_path / 'a',), version='different')
    # D4: code metadata and sample payload content are not replay requirements.
    a['code'] = {'commit': 'unrelated', 'dirty': False}
    (tmp_path / 'a/manifest.json').write_text(json.dumps(a))
    for path in (tmp_path / 'a/train').glob('*.json'):
        path.unlink()
    assert len(load_references((tmp_path / 'a',), version='test')) == 1
    a['splits']['train']['samples'][0]['sample_id'] = 'wrong'
    (tmp_path / 'a/manifest.json').write_text(json.dumps(a))
    with pytest.raises(ValueError, match='sample_id'):
        load_references((tmp_path / 'a',), version='test')
    legacy = tmp_path / 'legacy'
    legacy_manifest = _legacy_fixture(legacy)
    assert len(load_references((legacy,), version='fixture')) == sum(
        s['kept'] for s in legacy_manifest['splits'].values())
    # Plugin replay preserves isolated nodes, even though measurement requires reachability.
    module = 'gr_dataset_test_plugin'
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    (tmp_path / f'{module}.py').write_text(
        'from cfg_reducer.family_registry import register_family\n'
        'from cfg_reducer.generator_types import CFGShape\n'
        'class Plugin:\n'
        '    name = "dataset_toy"\n'
        '    def normalize(self, spec): return spec\n'
        '    def generate(self, spec, rng):\n'
        '        nodes = tuple(str(i) for i in range(spec.num_nodes))\n'
        '        edges = () if spec.params.get("isolated") else tuple(zip(nodes, nodes[1:]))\n'
        '        return CFGShape(nodes, edges, nodes[0])\n'
        'register_family(Plugin())\n')
    spec.write_text(json.dumps({'family': 'dataset_toy', 'num_nodes': 2, 'params': {}}))
    assert run('plugin', '--plugin', module)['generator']['name'] == 'cfg_v2:dataset_toy'
    assert len(load_references((tmp_path / 'plugin',), version='test')) == 1
    from cfg_reducer.generate_v2 import descriptor_for, spec_from_json, spec_to_json
    isolated_spec = spec_from_json({'family': 'dataset_toy', 'num_nodes': 2, 'params': {'isolated': True}})
    build_dataset(tmp_path / 'isolated', {'train': (0, 1)}, {'spec': spec_to_json(isolated_spec)},
                  'test', descriptor_for(isolated_spec))
    refs = load_references((tmp_path / 'isolated',), version='test')
    assert refs[0].nodes == ('0', '1') and refs[0].edges == ()
    excluded = build_dataset(tmp_path / 'isolated_b', {'test': (2, 3)},
                             {'spec': spec_to_json(isolated_spec)}, 'test',
                             descriptor_for(isolated_spec), exclude=refs)
    assert excluded['splits']['test']['rejected_by_reason'] == {'cross_dataset_duplicate': 1}


def test_manifest_hashseed_determinism(tmp_path):
    import os
    import subprocess
    import sys

    script = '''
import json, sys
from pathlib import Path
from cfg_reducer.dataset_v2 import main
root = Path(sys.argv[1])
root.mkdir()
for family in ('layered', 'structured', 'spaghetti'):
    spec = root / 'spec.json'
    spec.write_text(json.dumps({'family': family, 'num_nodes': 12, 'params': {}}))
    for target in (1, 2):
        plan = root / 'plan.json'
        plan.write_text(json.dumps({'train': {
            'dims': [{'feature': 'num_nodes', 'cuts': [], 'labels': ['all']}],
            'target_per_bucket': target, 'ranges': {'num_nodes': [None, 20]}}}))
        main(['--spec', str(spec), '--plan', str(plan), '--out', str(root / f'{family}_{target}'),
              '--split', 'train=0:5', '--version', 'determinism', '--allow-incomplete'])
'''
    for seed in ('1', '77'):
        subprocess.run([sys.executable, '-c', script, str(tmp_path / seed)],
                       env=os.environ | {'PYTHONHASHSEED': seed}, check=True)
    def artifacts(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.json*')}
    assert artifacts(tmp_path / '1') == artifacts(tmp_path / '77')
    for family in ('layered', 'structured', 'spaghetti'):
        manifests = [json.loads((tmp_path / '1' / f'{family}_{target}' / 'manifest.json').read_bytes())
                     for target in (1, 2)]
        small, large = [{s['seed']: s['sample_id'] for s in m['splits']['train']['samples']}
                        for m in manifests]
        assert small and small.items() <= large.items()
        assert all(m['rejected'] > 0 for m in manifests)
