"""Synthetic, torch-free acceptance tests for P3 aggregation."""
from __future__ import annotations

import hashlib
import json
from math import log
from pathlib import Path

import pytest

from cfg_reducer.model_input import build_vocab
from experiments.pretrain.make_plan import make_plan
from experiments.pretrain.make_spec import params_for_n
from training import eval_buckets as eb


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def entry(sid: str, n: int, seed: int = 0, family: str = 'layered') -> dict:
    params = params_for_n(n) if family == 'mixture' else next(c['params'] for c in params_for_n(n)['components'] if c['family'] == family)
    return dict(sample_id=sid, seed=seed, requested={'spec': dict(family=family, num_nodes=n, params=params)})


def manifest(splits: dict) -> dict:
    return {'generator': {'version': 'test-v1', 'config': {'spec': {'family': 'layered', 'num_nodes': 48, 'params': {}}}},
            'splits': {split: {'samples': rows} for split, rows in splits.items()}}


def score(row: dict, loss: float) -> dict:
    count = len(row['tokens']) - 1
    return dict(sample_id=row['sample_id'], seed=row['seed'], n_tokens=count, nll=count * loss,
                nll_per_token=loss, token_nll=[loss] * count)


@pytest.fixture
def fixture(tmp_path):
    _, index = make_plan('compare')
    # Use two models to exercise token-weighted pairing; paths carry no n/family hints.
    index['runs'] = index['runs'][:2]
    vocab = build_vocab(128)
    short = [vocab['BOS'], vocab['KIND_ENTRY'], vocab['EOS']]
    long = [vocab['BOS'], vocab['KIND_ENTRY'], vocab['KIND_LINEAR'], vocab['EOS']]
    train_entries = [entry(f'train-{n}', n) for n in eb.ID_NS]
    train = [dict(sample_id=r['sample_id'], seed=r['seed'], tokens=short) for r in train_entries]
    source = tmp_path / index['source_dataset']
    write_json(source / 'manifest.json', manifest({'train': train_entries, 'val': []}))
    for i, bucket in enumerate(index['buckets']):
        n = bucket['n']
        bucket['dataset'] = f'data/opaque-{i}'
        dataset = tmp_path / bucket['dataset']
        raw = [entry(f'a-{i}', n), entry(f'b-{i}', n, family='structured'), entry(f'excluded-{i}', n)]
        write_json(dataset / 'manifest.json', manifest({'test': raw}))
        tests = [dict(sample_id=raw[0]['sample_id'], seed=0, tokens=short),
                 dict(sample_id=raw[1]['sample_id'], seed=0, tokens=long)]
        bundle = tmp_path / bucket['bundle']
        write_json(bundle / 'vocab.json', vocab)
        write_json(bundle / 'meta.json', {'max_offset': 128})
        write_rows(bundle / 'train.jsonl', train)
        write_rows(bundle / 'val.jsonl', [])
        write_rows(bundle / 'test.jsonl', tests)
        write_json(bundle / 'evaluation_index.json', {
            'version': 1, 'sources': {'train': eb.digest(source / 'manifest.json'),
                                     'val': eb.digest(source / 'manifest.json'), 'test': eb.digest(dataset / 'manifest.json')},
            'samples': {r['sample_id']: {'split': split} for split, rows in [('train', train), ('test', tests)] for r in rows},
            'exclusions': [dict(sample_id=raw[2]['sample_id'], seed=0, split='test', reason='over_window', needed=129)]})
        for j, run in enumerate(index['runs']):
            write_rows(tmp_path / run['scores_by_n'][str(n)], [score(tests[1], 2 + j), score(tests[0], 1 + j)])
    for run in index['runs']:
        path = tmp_path / run['source_run']
        write_rows(path / 'test_scores.jsonl', eb.read_jsonl(tmp_path / run['scores_by_n']['48']))
        # Include failed generations, including a clean prefix without EOS.
        write_json(path / 'samples.json', {'samples': [short, short[:-1], [vocab['BOS'], vocab['REF_1'], vocab['EOS']]]})
        write_json(path / 'samples_constrained.json', {'samples': [short, short[:-1]]})
    return index, vocab


def test_hand_computed_metrics_and_baseline(tmp_path, fixture):
    index, _ = fixture
    report = eb.evaluate(index, tmp_path)
    run = report['runs'][index['runs'][0]['name']]
    row = run['by_n']['48']
    assert (row['raw'], row['retained'], row['excluded'], row['tokens']) == (3, 2, 1, 5)
    assert row['nats_per_token'] == pytest.approx(8 / 5)
    # One training stream BOS ENTRY EOS. Seven structural tokens; add-alpha.
    # Short: 2 * [-log(.75) -log(1.5/4.5)].
    # Long: ENTRY as above; LINEAR -log(.75)-log(.5/4.5);
    # EOS unseen type context -log(.5), unigram -log(1.5/5.5).
    short_nll = -2 * log(.75 * 1.5 / 4.5)
    long_nll = -log(.75 * 1.5 / 4.5) - log(.75 * .5 / 4.5) - log(.5 * 1.5 / 5.5)
    assert row['baseline_nats_per_token'] == pytest.approx((short_nll + long_nll) / 5)
    assert row['excess_nats_per_token'] == pytest.approx((8 - short_nll - long_nll) / 5)
    assert row['by_family']['layered']['raw'] == 2
    assert row['by_family']['layered']['tokens'] == 2
    assert row['by_family']['structured']['nats_per_token'] == 2
    assert run['by_family']['layered']['id']['tokens'] == 18
    assert run['id']['token_micro'] == pytest.approx(1.6)
    assert run['id']['macro_excluded_ns'] == [8, 12]
    assert report['baselines']['48']['fit_samples'] == 1
    assert report['baselines']['192']['fit_samples'] == 9
    assert report['baselines']['192']['baseline_fit_scope'] == 'pooled_id'
    assert report['paired'][0]['by_n']['48']['ci95'] == [-1, -1]
    wf = run['unconditional_wf']['reference-constrained']
    assert wf['total'] == 3 and wf['well_formed_rate'] == pytest.approx(1 / 3)
    assert wf['violations']['no_eos:would_close_cleanly'] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nan', 'tokens', 'count', 'hash', 'missing_file', 'missing_sample_file', 'duplicate_n', 'duplicate_run'])
def test_reject_corruption(tmp_path, fixture, damage):
    index, _ = fixture
    path = tmp_path / index['runs'][0]['scores_by_n']['48']
    rows = eb.read_jsonl(path)
    if damage == 'missing':
        write_rows(path, rows[:1])
    elif damage == 'duplicate':
        write_rows(path, rows + rows[:1])
    elif damage == 'nan':
        rows[0]['nll'] = float('nan')
        write_rows(path, rows)
    elif damage == 'count':
        rows[0]['n_tokens'] += 1
        write_rows(path, rows)
    elif damage == 'tokens':
        token_path = tmp_path / index['source_bundle'] / 'test.jsonl'
        write_rows(token_path, eb.read_jsonl(token_path) * 2)
    elif damage == 'hash':
        p = tmp_path / index['source_bundle'] / 'evaluation_index.json'
        data = eb.read_json(p)
        data['sources']['test'] = 'wrong'
        write_json(p, data)
    elif damage == 'missing_file':
        path.unlink()
    elif damage == 'missing_sample_file':
        (tmp_path / index['runs'][0]['source_run'] / 'samples_constrained.json').unlink()
    elif damage == 'duplicate_n':
        index['buckets'].append(index['buckets'][0])
    else:
        index['runs'].append(index['runs'][0])
    with pytest.raises((ValueError, FileNotFoundError)):
        eb.evaluate(index, tmp_path)


def test_mixture_family_from_resolved_manifest(tmp_path):
    from cfg_reducer.families.mixture import component_for
    from cfg_reducer.generate_v2 import spec_from_json
    raw = [entry('opaque-a', 48, 0, 'mixture'), entry('opaque-b', 192, 5, 'mixture')]
    write_json(tmp_path / 'manifest.json', manifest({'test': raw}))
    result = eb.manifest_rows(tmp_path, 'test')
    for row in raw:
        spec = row['requested']['spec']
        family = spec['params']['components'][component_for(spec_from_json(spec), row['seed'])]['family']
        assert result[row['sample_id']]['family'] == family
        assert result[row['sample_id']]['n'] == spec['num_nodes']


def test_wrong_k_is_not_legality_violation():
    v = build_vocab(128)
    tokens = [v['BOS'], v['KIND_ENTRY'], v['KIND_LINEAR'], v['KIND_LINEAR'], v['REF_1'], v['EOS']]
    row = score(dict(sample_id='a', seed=0, tokens=tokens), 2)
    row.update(ref_pos=[3], ref_k=[1], ref_correct=[0], ref_type_nll=[.5], ref_pred_k=[2], ref_pred_legal=[True])
    ce_tokens = {'a': tokens}
    eb.ce.validate_scores([row], ce_tokens, expect_ids={'a'})
    report = eb.ref_metrics([row], ce_tokens, v)
    assert report['edge_accuracy']['overall'] == 0
    assert report['gold_ref_by_k'][1]['teacher_forced_legality']['violation_rate'] == 0
    assert report['offset_nll_by_k'][1]['mean'] == 1.5
    row.update(ref_pred_k=[3], ref_pred_legal=[False])
    assert eb.ref_metrics([row], ce_tokens, v)['gold_ref_by_k'][1]['teacher_forced_legality']['violation_rate'] == 1
    row['ref_pred_legal'] = [True]
    with pytest.raises(ValueError, match='legality'):
        eb.ref_metrics([row], ce_tokens, v)


def test_continuation_stops_at_first_violation():
    v = build_vocab(128)
    gold = [v['BOS'], v['KIND_ENTRY'], v['KIND_LINEAR'], v['REF_1'], v['EOS']]
    seed = int.from_bytes(hashlib.sha256(b'0:a:wf-v1').digest()[:8], 'big')
    prefix = gold[:2]
    bad = prefix + [v['REF_1'], v['REF_128'], v['EOS']]
    row = {'sample_id': 'a', 'wf_probe': {'prefix_len': 2, 'budget': 10, 'seed': seed,
                                         'raw': bad, 'constrained': gold}}
    report = eb.continuation([row], {'a': gold}, v, 192, 0)
    assert report['prefix_source_n'] == 192
    raw = report['reference-constrained']
    assert raw['total'] == 1 and raw['well_formed'] == 0
    assert raw['stopped_at_first_violation'] == 1
    assert raw['suffix_ref_by_predicted_k'] == {1: {'refs': 1, 'violations': 1, 'violation_rate': 1}}
    assert report['grammar-constrained']['well_formed_rate'] == 1
    empty = eb.continuation([], {}, v, 256, 0)
    assert empty['reference-constrained']['ci95'] is None


def test_token_weighted_paired_ci():
    left = [dict(sample_id='a', n_tokens=1, nll=4), dict(sample_id='b', n_tokens=3, nll=0)]
    right = [dict(sample_id='b', n_tokens=3, nll=0), dict(sample_id='a', n_tokens=1, nll=0)]
    result = eb.paired_ci(left, right)
    assert result['delta_nats_per_token'] == 1  # Not the sample mean of 2.
    assert result['ci95'] == [0, 4]
    assert eb.paired_ci(list(reversed(left)), right) == result


def test_empty_bucket_and_no_fit(tmp_path, fixture):
    index, _ = fixture
    bucket = next(b for b in index['buckets'] if b['n'] == 48)
    bundle = tmp_path / bucket['bundle']
    data = eb.read_json(bundle / 'evaluation_index.json')
    test_rows = eb.read_jsonl(bundle / 'test.jsonl')
    for r in test_rows:
        del data['samples'][r['sample_id']]
        data['exclusions'].append(dict(sample_id=r['sample_id'], seed=r['seed'], split='test', reason='over_window'))
    write_json(bundle / 'evaluation_index.json', data)
    write_rows(bundle / 'test.jsonl', [])
    for run in index['runs']:
        write_rows(tmp_path / run['scores_by_n']['48'], [])
        write_rows(tmp_path / run['source_run'] / 'test_scores.jsonl', [])
    result = eb.evaluate(index, tmp_path)['runs'][index['runs'][0]['name']]
    assert result['by_n']['48']['raw'] == 3
    assert result['by_n']['48']['retained'] == 0
    assert result['by_n']['48']['nats_per_token'] is None
    assert result['by_n']['48']['excess_nats_per_token'] is None
    assert result['id']['bucket_macro'] is None


def test_seed_summary_and_cli(tmp_path, fixture, monkeypatch):
    index, _ = fixture
    second = index['runs'][1]
    second['pos'] = index['runs'][0]['pos']
    second['seed'] = 1
    report = eb.evaluate(index, tmp_path)
    stats = report['summary'][second['pos']]['by_n']['48']['nats_per_token']
    assert stats['per_seed'] == {'0': 1.6, '1': 2.6}
    assert stats['mean'] == pytest.approx(2.1)
    assert stats['sd'] == pytest.approx(.5)
    write_json(tmp_path / 'index.json', index)
    monkeypatch.chdir(tmp_path)
    eb.main(['--index', 'index.json', '--out', 'report.json'])
    assert eb.read_json(tmp_path / 'report.json')['schema_version'] == 1
    assert '長系列' in (tmp_path / 'report.md').read_text()


def test_missing_id_training_fit_is_null(tmp_path, fixture):
    index, _ = fixture
    for bucket in index['buckets']:
        bundle = tmp_path / bucket['bundle']
        write_rows(bundle / 'train.jsonl', [r for r in eb.read_jsonl(bundle / 'train.jsonl') if r['sample_id'] != 'train-48'])
        data = eb.read_json(bundle / 'evaluation_index.json')
        del data['samples']['train-48']
        data['exclusions'].append(dict(sample_id='train-48', seed=0, split='train', reason='over_window'))
        write_json(bundle / 'evaluation_index.json', data)
    report = eb.evaluate(index, tmp_path)
    assert report['baselines']['48']['fit_samples'] == 0
    row = report['runs'][index['runs'][0]['name']]['by_n']['48']
    assert row['excess_nats_per_token'] is None
    assert row['nats_per_token'] == 1.6


@pytest.mark.parametrize('damage', [None, 'vocab', 'train', 'source_hash', 'version'])
def test_plan_checks_present_bundles(tmp_path, fixture, monkeypatch, damage):
    from experiments.pretrain.make_plan import main
    index, _ = fixture
    for bucket in index['buckets']:
        dest = tmp_path / f"data/pretrain_test_n{bucket['n']}"
        dest.mkdir()
        (dest / 'manifest.json').write_bytes((tmp_path / bucket['dataset'] / 'manifest.json').read_bytes())
    target = tmp_path / 'data/tok_pretrain_n192'
    if damage == 'vocab':
        write_json(target / 'vocab.json', {})
    elif damage == 'train':
        path = target / 'train.jsonl'
        write_rows(path, list(reversed(eb.read_jsonl(path))))
    elif damage == 'source_hash':
        data = eb.read_json(target / 'evaluation_index.json')
        data['sources']['val'] = 'bad'
        write_json(target / 'evaluation_index.json', data)
    elif damage == 'version':
        path = tmp_path / 'data/pretrain_test_n192/manifest.json'
        data = eb.read_json(path)
        data['generator']['version'] = 'another-commit'
        write_json(path, data)
    monkeypatch.chdir(tmp_path)
    if damage:
        with pytest.raises(ValueError):
            main(['--phase', 'compare', '--out-dir', 'plans', '--allow-missing-bundles'])
        assert not (tmp_path / 'plans').exists()
    else:
        main(['--phase', 'compare', '--out-dir', 'plans'])
        assert (tmp_path / 'plans/plan_compare.json').exists()


@pytest.mark.parametrize('damage', ['probe', 'sample', 'source_rescore'])
def test_run_artifact_completeness(tmp_path, fixture, damage):
    index, _ = fixture
    run = index['runs'][0]
    path = tmp_path / run['source_run']
    payload = eb.read_json(path / 'samples.json')
    payload['config'] = {'seed': run['seed'], 'pos': run['pos'], 'ref_legal_mask': True,
                         'wf_probes': 1 if damage == 'probe' else 0,
                         'num_samples': 4 if damage == 'sample' else 3, 'constrained_samples': 2}
    write_json(path / 'samples.json', payload)
    if damage == 'source_rescore':
        rows = eb.read_jsonl(path / 'test_scores.jsonl')
        rows[0]['token_nll'] = [3.] * rows[0]['n_tokens']
        rows[0]['nll'] = 3. * rows[0]['n_tokens']
        rows[0]['nll_per_token'] = 3.
        write_rows(path / 'test_scores.jsonl', rows)
    with pytest.raises(ValueError, match={'probe': 'wf_probe', 'sample': 'generated samples', 'source_rescore': 'n48 rescore'}[damage]):
        eb.evaluate(index, tmp_path)


def test_train_length_counts_are_reported(tmp_path, fixture):
    index, _ = fixture
    run = index['runs'][0]
    stats = {'raw': {'samples': 9, 'tokens': 18}, 'kept': {'samples': 9, 'tokens': 18},
             'dropped': {'samples': 0, 'tokens': 0}, 'effective_max_len': 3, 'excluded': []}
    write_json(tmp_path / run['source_run'] / 'length_stats.json', stats)
    assert eb.evaluate(index, tmp_path)['runs'][run['name']]['train_length_stats'] == stats
