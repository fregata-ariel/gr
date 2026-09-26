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
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(tmp_path.rglob('*.json')):
        digest.update(str(path.relative_to(tmp_path)).encode())
        digest.update(path.read_bytes())
    # Captured from HEAD's builder before P1; includes manifest and sample bytes.
    assert digest.hexdigest() == '508cb670572c4671bdee811d032d76459e641f55cf62b7093cd2ce7969a2d0ad'
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


def test_load_references_version_mismatch_needs_explicit_override(tmp_path):
    from cfg_reducer import dataset_v2
    from cfg_reducer.generate_v2 import descriptor_for, normalize_spec, spec_to_json
    from cfg_reducer.generator_types import GeneratorSpec

    spec = GeneratorSpec("layered", 8, {"edge_prob": 0.3})
    build_dataset(tmp_path / "a", {"train": (0, 3)},
                          {"spec": spec_to_json(normalize_spec(spec))}, "v1",
                          descriptor_for(spec))
    with pytest.raises(ValueError, match="version mismatch"):
        dataset_v2.load_references((tmp_path / "a",), version="v2")
    refs = dataset_v2.load_references((tmp_path / "a",), version="v2",
                                      allow_version_mismatch=True)
    assert len(refs) == 3 and all(r.nodes and r.edges for r in refs)
    assert dataset_v2.load_references((tmp_path / "a",), version="v1") == refs


@pytest.mark.parametrize('accepted', [False, True])
def test_nonfinite_realized_is_strict_json_rejection(tmp_path, accepted):
    features = {'mean_offset': float('nan'), 'max_depth': float('inf'),
                'min_offset': -float('inf'), 'num_nodes': 1}

    def accept(candidate, state):
        return AcceptDecision(accepted, None if accepted else 'invalid_feature', None, features)

    manifest = build_dataset(tmp_path, {'test': (0, 1)}, {}, 'test', toy, accept=accept)
    text = (tmp_path / 'rejections.jsonl').read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    row = json.loads(text)
    assert row['reason'] == 'invalid_feature'
    assert row['invalid_features'] == ['max_depth', 'mean_offset', 'min_offset']
    assert row['realized'] == dict(mean_offset=None, max_depth=None, min_offset=None, num_nodes=1)
    assert features['max_depth'] == float('inf')  # Hook-owned values are not mutated.
    assert manifest['splits']['test']['samples'] == []
    assert manifest['rejected'] == 1
    json.dumps(manifest, allow_nan=False)


def distribution(**updates):
    return {'family': 'layered', 'num_nodes': {'choices': [8, 12], 'weights': [1, 2]},
            'params': {}} | updates


@pytest.mark.parametrize('patch', [
    {'extra': 1}, {'params_by_num_nodes': {'8': {}, '12': {}}},
    {'num_nodes': {'choices': [], 'weights': []}},
    {'num_nodes': {'choices': [12, 8], 'weights': [1, 1]}},
    {'num_nodes': {'choices': [8, 8], 'weights': [1, 1]}},
    {'num_nodes': {'choices': [True], 'weights': [1]}},
    {'num_nodes': {'choices': [0], 'weights': [1]}},
    {'num_nodes': {'choices': [8.0], 'weights': [1]}},
    {'num_nodes': {'choices': [8], 'weights': [True]}},
    {'num_nodes': {'choices': [8], 'weights': [0]}},
    {'num_nodes': {'choices': [8], 'weights': [-1]}},
    {'num_nodes': {'choices': [8], 'weights': [float('inf')]}},
    {'num_nodes': {'choices': [8], 'weights': [float('nan')]}},
    {'num_nodes': {'choices': [8], 'weights': []}},
    {'num_nodes': {'choices': [8], 'weights': [1], 'extra': 0}},
    {'num_nodes': {'choices': '8', 'weights': [1]}},
    {'params': {'unknown': 1}},
])
def test_distribution_validation(patch):
    from cfg_reducer.dataset_v2 import dataset_spec_from_json
    with pytest.raises(ValueError):
        dataset_spec_from_json(distribution(**patch))


@pytest.mark.parametrize('params', [{'8': {}}, {'08': {}, '12': {}}, {'8': {}, '12': {}, '16': {}},
                                    {'8': {}, '12': {'unknown': 1}}, []])
def test_distribution_parameter_keys(params):
    from cfg_reducer.dataset_v2 import dataset_spec_from_json
    value = distribution()
    del value['params']
    value['params_by_num_nodes'] = params
    with pytest.raises(ValueError):
        dataset_spec_from_json(value)
    value['num_nodes'] = 8
    with pytest.raises(ValueError):
        dataset_spec_from_json(value)


def test_resolver_contract():
    import math
    from random import Random
    from cfg_reducer.dataset_v2 import dataset_spec_from_json, dataset_spec_to_json, resolve_sample_config
    from cfg_reducer.generate_v2 import spec_from_json
    fixed: dict = {'spec': {'family': 'layered', 'num_nodes': 8, 'params': {}}}
    assert resolve_sample_config(fixed, 1) is fixed
    assert dataset_spec_from_json(fixed['spec']) == spec_from_json(
        {'family': 'layered', 'num_nodes': 8, 'params': {}})
    config = {'spec': distribution(num_nodes={'choices': [8, 12], 'weights': [1e308, 1e308]})}
    parsed = dataset_spec_from_json(config['spec'])
    assert dataset_spec_from_json(dataset_spec_to_json(parsed)) == parsed
    forward = {s: resolve_sample_config(config, s) for s in range(20)}
    assert forward == {s: resolve_sample_config(config, s) for s in reversed(range(20))}
    for seed, resolved in forward.items():
        draw = Random('gr:num_nodes:v1:' + str(seed)).random() * math.fsum((1, 1))
        assert resolved['spec']['num_nodes'] == (8 if draw < 1 else 12)


def test_distribution_uuid_replay_and_composition(tmp_path):
    from cfg_reducer.dataset_v2 import main, load_references, resolve_sample_config
    from cfg_reducer.generate_v2 import descriptor_for, spec_from_json
    from cfg_reducer.families.mixture import component_for
    from experiments.pretrain.make_spec import params_for_n
    from training.mixture_doe import realized_composition
    value = distribution(family='mixture')
    del value['params']
    value['params_by_num_nodes'] = {str(n): params_for_n(n) for n in (8, 12)}
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(value))
    out = tmp_path / 'mix'
    main(['--spec', str(spec_path), '--out', str(out), '--split', 'train=0:8', '--version', 'test'])
    manifest = json.loads((out / 'manifest.json').read_bytes())
    samples = manifest['splits']['train']['samples']
    expected = dict(layered=0, structured=0, spaghetti=0)
    for sample in samples:
        resolved = resolve_sample_config({'spec': value}, sample['seed'])
        spec = spec_from_json(resolved['spec'])
        components = spec.params['components']
        assert isinstance(components, list)
        component = components[component_for(spec, sample['seed'])]
        assert isinstance(component, dict)
        family = component['family']
        assert isinstance(family, str)
        expected[family] += 1
        fixed = build_dataset(tmp_path / f"fixed{sample['seed']}", {'test': (sample['seed'], sample['seed'] + 1)},
                              resolved, 'test', descriptor_for(spec))
        assert fixed['splits']['test']['samples'][0]['sample_id'] == sample['sample_id']
        assert sample['requested'] == resolved
        assert sample['bucket'] == f'n{spec.num_nodes}'
        assert sample['realized']['num_nodes'] == spec.num_nodes
    assert realized_composition(out) == expected
    assert len(load_references((out,), version='test')) == len(samples)
    for sample in samples:
        spec = spec_from_json(sample['requested']['spec'])
        reordered = build_dataset(tmp_path / f"reordered{sample['seed']}",
            {'different_split': (sample['seed'], sample['seed'] + 1)}, {'spec': value},
            'test', descriptor_for(spec), resolve_config=resolve_sample_config, record_node_counts=True)
        assert reordered['splits']['different_split']['samples'][0]['sample_id'] == sample['sample_id']
    original_id = samples[0]['sample_id']
    samples[0]['sample_id'] = 'wrong'
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='sample_id'):
        load_references((out,), version='test')
    samples[0]['sample_id'] = original_id
    (out / 'manifest.json').write_text(json.dumps(manifest))
    # A weight change alters the manifest but not identity for unchanged resolutions.
    value['num_nodes']['weights'] = [2, 3]
    spec_path.write_text(json.dumps(value))
    main(['--spec', str(spec_path), '--out', str(tmp_path / 'changed'), '--split', 'train=0:8', '--version', 'test'])
    changed = json.loads((tmp_path / 'changed/manifest.json').read_bytes())
    shared = 0
    for sample in changed['splits']['train']['samples']:
        previous = next((s for s in samples if s['seed'] == sample['seed'] and s['requested'] == sample['requested']), None)
        if previous:
            shared += 1
            assert previous['sample_id'] == sample['sample_id']
    assert shared
    sample = samples[0]
    payload_path = out / 'train' / f"{sample['sample_id']}.json"
    payload = json.loads(payload_path.read_bytes())
    payload['provenance']['generator']['seed'] += 1
    payload_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='provenance'):
        load_references((out,), version='test')
    sample['requested'] = {}
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='requested'):
        load_references((out,), version='test')


def test_node_stats_and_dedup_across_nominal_counts(tmp_path):
    calls = []
    def resolve(config, seed):
        calls.append(seed)
        return {'spec': {'num_nodes': 8 if seed % 2 == 0 else 12}}
    def generate(engine, *, seed, spec):
        return toy(engine, seed=seed)
    result = build_dataset(tmp_path, {'train': (0, 7)},
        {'spec': {'num_nodes': {'choices': [8, 12, 16]}}}, 'test', generate,
        resolve_config=resolve, record_node_counts=True)
    assert calls == list(range(7))
    split = result['splits']['train']
    assert split['rejected_by_reason'] == {'duplicate': 2, 'node_budget': 1}
    assert split['per_num_nodes'] == {
        '8': dict(attempts=4, generated=4, accepted=4, rejected=0, rejected_by_reason={},
                  cfg_num_nodes_histogram={'1': 1, '2': 1, '3': 1, '4': 1}),
        '12': dict(attempts=3, generated=2, accepted=0, rejected=3,
                   rejected_by_reason={'duplicate': 2, 'node_budget': 1}, cfg_num_nodes_histogram={}),
        '16': dict(attempts=0, generated=0, accepted=0, rejected=0,
                   rejected_by_reason={}, cfg_num_nodes_histogram={})}
    assert all(json.loads(line)['num_nodes'] == 12 for line in (tmp_path / 'rejections.jsonl').read_text().splitlines())
    refs = (CFGReference('other', 'different_uuid', ('a',), ()),)
    excluded = build_dataset(tmp_path / 'excluded', {'test': (1, 2)},
        {'spec': {'num_nodes': 12}}, 'test', generate, exclude=refs, record_node_counts=True)
    assert excluded['splits']['test']['rejected_by_reason'] == {'cross_dataset_duplicate': 1}


def test_target_counts_cli(tmp_path):
    from cfg_reducer.dataset_v2 import main
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'family': 'layered', 'num_nodes': 8, 'params': {}}))
    def run(name, *flags):
        main(['--spec', str(spec), '--out', str(tmp_path / name), '--split', 'train=0:4',
              '--split', 'val=10:11', '--version', 'test', *flags])
        return json.loads((tmp_path / name / 'manifest.json').read_bytes())['splits']
    split = run('complete', '--target-count', 'train=1')['train']
    assert (split['attempts'], split['last_seed'], split['target'], split['missing'], split['complete']) == (1, 0, 1, 0, True)
    assert split['seed_range'] == [0, 4]
    with pytest.raises(SystemExit) as exc:
        run('missing', '--target-count', 'train=10')
    assert exc.value.code == 2
    split = run('allowed', '--target-count', 'train=10', '--allow-incomplete')['train']
    assert split['attempts'] == 4 and split['last_seed'] == 3
    assert split['missing'] == 10 - split['accepted'] and not split['complete']
    for flags in [('--target-count', 'unknown=1'), ('--target-count', 'train=0'),
                  ('--target-count', 'train=-1'), ('--target-count', 'train=1.5'),
                  ('--target-count', 'train=1', '--target-count', 'train=2'),
                  ('--target-count', 'train=1', '--plan', 'unused.json')]:
        with pytest.raises(ValueError):
            run('invalid', *flags)
    spec.write_text(json.dumps(distribution()))
    with pytest.raises(ValueError, match='--plan'):
        run('invalid', '--plan', 'unused.json')


def test_distribution_hashseed(tmp_path):
    import os
    import subprocess
    import sys
    script = '''
import sys, json
from pathlib import Path
from cfg_reducer.dataset_v2 import main
root = Path(sys.argv[1]); root.mkdir()
spec = root / 'spec.json'
spec.write_text(json.dumps({'family': 'layered', 'num_nodes': {'choices': [8,12], 'weights': [1,2]}, 'params': {}}))
main(['--spec', str(spec), '--out', str(root / 'data'), '--split', 'train=0:6', '--version', 'test'])
'''
    for seed in ('1', '77'):
        subprocess.run([sys.executable, '-c', script, str(tmp_path / seed)],
                       env=os.environ | {'PYTHONHASHSEED': seed}, check=True)
    def artifacts(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.json*')}
    assert artifacts(tmp_path / '1') == artifacts(tmp_path / '77')
