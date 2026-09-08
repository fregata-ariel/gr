"""controlled_eval: interval statistics, paired deltas, REF records and
frequency baselines on synthetic token streams (torch-free)."""

import math

import pytest

from cfg_reducer import model_input
from training import controlled_eval as ce


def test_wilson_interval_brackets_the_rate_and_shrinks_with_n():
    lo, hi = ce.wilson(90, 100)
    assert lo < 0.9 < hi
    lo2, hi2 = ce.wilson(900, 1000)
    assert hi2 - lo2 < hi - lo
    assert ce.wilson(0, 0) == (0.0, 0.0)
    assert ce.wilson(0, 10)[0] == 0.0 and ce.wilson(10, 10)[1] == 1.0


def test_bootstrap_ci_contains_the_mean_and_is_deterministic():
    values = [float(i) for i in range(50)]
    lo, hi = ce.bootstrap_ci(values, n_boot=500, seed=1)
    assert lo < sum(values) / len(values) < hi
    assert ce.bootstrap_ci(values, n_boot=500, seed=1) == (lo, hi)


def test_paired_delta_joins_on_sample_id():
    base = [{"sample_id": "a", "nll_per_token": 1.0},
            {"sample_id": "b", "nll_per_token": 2.0},
            {"sample_id": "zzz", "nll_per_token": 9.0}]
    conf = [{"sample_id": "b", "nll_per_token": 1.5},
            {"sample_id": "a", "nll_per_token": 1.2}]
    out = ce.paired_delta(conf, base)
    assert out["n"] == 2
    assert out["mean_delta"] == pytest.approx((0.2 - 0.5) / 2)
    assert out["frac_improved"] == 0.5
    with pytest.raises(ValueError):
        ce.paired_delta([{"sample_id": "q", "nll_per_token": 0.0}], base)


def _stream(vocab, names):
    return [vocab[n] for n in names]


def test_ref_records_capture_k_n_legal_rel_and_edge_correctness():
    vocab = model_input.build_vocab(3)
    names = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_LINEAR", "KIND_MERGE",
             "REF_1", "REF_3", "EOS"]
    tokens = _stream(vocab, names)
    # targets are tokens[1:]; REF_1 is target index 4, REF_3 is index 5
    score = {"sample_id": "s", "token_nll": [0.1] * 4 + [0.5, 1.5, 0.2],
             "ref_pos": [4, 5], "ref_k": [1, 3], "ref_correct": [1, 0],
             "ref_type_nll": [0.1, 0.5]}
    recs = ce.ref_records({"s": tokens}, [score], vocab, max_k=3)
    assert [r["k"] for r in recs] == [1, 3]
    # after KIND_MERGE (4th motif): 3 earlier motifs, window 3 -> 3 legal
    assert recs[0]["n_legal"] == 3 and recs[0]["rel"] == 1
    # after REF_1: legal k in {2, 3} -> 2 legal, rel = 3 - 1
    assert recs[1]["n_legal"] == 2 and recs[1]["rel"] == 2
    assert [r["correct"] for r in recs] == [1, 0]
    assert [r["nll"] for r in recs] == [0.5, 1.5]
    assert [r["offset_nll"] for r in recs] == pytest.approx([0.4, 1.0])
    assert ce.offset_by_k(recs, min_count=1) == {1: {"n": 1, "mean": pytest.approx(0.4)},
                                                 3: {"n": 1, "mean": pytest.approx(1.0)}}
    # score files without ref_type_nll degrade gracefully
    bare = ce.ref_records({"s": tokens}, [{k: v for k, v in score.items() if k != "ref_type_nll"}],
                          vocab, max_k=3)
    assert all(r["offset_nll"] is None for r in bare) and ce.offset_by_k(bare) == {}

    acc = ce.edge_accuracy(recs)
    assert acc["overall"] == 0.5 and acc["n"] == 2
    assert acc["by_k"] == {1: {"n": 1, "acc": 1.0}, 3: {"n": 1, "acc": 0.0}}


def test_by_k_macro_micro_and_strata():
    recs = ([{"k": 1, "nll": 0.2, "n_legal": 2, "rel": 1, "correct": None}] * 30
            + [{"k": 2, "nll": 1.0, "n_legal": 2, "rel": 2, "correct": None}] * 10
            + [{"k": 2, "nll": 3.0, "n_legal": 5, "rel": 2, "correct": None}] * 5)
    table = ce.by_k(recs)
    assert table[1]["n"] == 30 and "ci95" in table[1]
    assert table[2]["n"] == 15 and table[2]["mean"] == pytest.approx(5 / 3)
    mm = ce.macro_micro(table)
    assert mm["macro"] == pytest.approx((0.2 + 5 / 3) / 2)
    assert mm["micro"] == pytest.approx((30 * 0.2 + 15 * 5 / 3) / 45)
    strata = ce.within_n_legal(recs)
    assert set(strata[2]) == {1, 2}          # both k have >= 10 records
    assert 5 not in strata or strata[5] == {}  # 5 records fall under min_count


def test_frequency_baselines_fit_on_train_refs():
    vocab = model_input.build_vocab(3)
    # every stream: three motifs then a merge with REF_1 REF_2 -> k=1 (rel 1,
    # n_legal 3) and k=2 (rel 1, n_legal 2)
    names = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_LINEAR", "KIND_MERGE",
             "REF_1", "REF_2", "EOS"]
    tokens = _stream(vocab, names)
    rows = [{"sample_id": f"t{i}", "tokens": tokens} for i in range(4)]
    fb = ce.FrequencyBaselines(rows, vocab, max_k=3, alpha=0.0)
    assert fb.total == 8
    assert fb.unigram_nll(1) == pytest.approx(-math.log(0.5))
    assert fb.unigram_nll(2) == pytest.approx(-math.log(0.5))
    # conditional: given n_legal 3 the only observed rel is 1 -> cost 0
    assert fb.conditional_nll(1, 3) == pytest.approx(0.0)
    smoothed = ce.FrequencyBaselines(rows, vocab, max_k=3, alpha=1.0)
    assert smoothed.conditional_nll(2, 3) > smoothed.conditional_nll(1, 3)
    recs = ce.ref_records({"t0": tokens},
                          [{"sample_id": "t0", "token_nll": [0.0] * 7}], vocab, 3)
    tables = smoothed.tables(recs, min_count=1)
    assert set(tables["unigram_nll_by_k"]) == {1, 2}
    assert set(tables["conditional_nll_by_k"]) == {1, 2}


def test_bucket_nll_and_pairing(tmp_path, capsys, monkeypatch):
    import json
    from training.data_utils import write_jsonl

    vocab = model_input.build_vocab(3)
    tokens = _stream(vocab, ['BOS', 'KIND_ENTRY', 'KIND_LINEAR', 'KIND_LINEAR',
                             'KIND_MERGE', 'REF_1', 'REF_3', 'EOS'])
    short = _stream(vocab, ['BOS', 'KIND_ENTRY', 'EOS'])
    index = {'version': 1, 'samples': {
        'a': {'split': 'test', 'bucket': 'x'},
        'b': {'split': 'test', 'bucket': 'x'},
        'c': {'split': 'test', 'bucket': None},
        'train': {'split': 'train', 'bucket': None}},
        'selection': {'plans': {'test': {'dims': [{'feature': 'depth', 'labels': ['empty']}],
                                        'active': None}}},
        'exclusions': [{'sample_id': 'excluded', 'split': 'test', 'bucket': 'lost'}]}
    scores: list[dict] = [dict(sample_id=sid, n_tokens=n, nll=n*v, nll_per_token=v,
                   token_nll=[v]*n, ref_pos=[4, 5] if sid != 'b' else [],
                   ref_correct=[1, 0] if sid != 'b' else [])
              for sid, n, v in [('a', 7, 1.), ('b', 2, 3.), ('c', 7, 2.)]]
    table = ce.bucket_scores(scores, index)
    assert table['x']['nll_per_token'] == pytest.approx(13/9)
    assert table['x']['nll_per_sample_mean'] == 2
    assert table['x']['ci95'] == list(ce.bootstrap_ci([1., 3.]))
    assert table['x']['ref_nll'] == 1 and table['x']['edge_accuracy'] == .5
    assert table['__unmeasured__']['n_samples'] == 1
    for name in ('lost', 'depth=empty'):
        assert table[name]['n_samples'] == table[name]['n_tokens'] == 0
        for key in ('nll_per_token', 'nll_per_sample_mean', 'ci95', 'ref_nll', 'edge_accuracy'):
            assert table[name][key] is None
    json.dumps(table, allow_nan=False)
    for bad in (scores[:-1], scores + scores[:1], scores + [dict(scores[0], sample_id='unknown')]):
        with pytest.raises(ValueError):
            ce.bucket_scores(bad, index)

    bundle, runs = tmp_path / 'tokens', tmp_path / 'runs'
    bundle.mkdir()
    (bundle / 'vocab.json').write_text(json.dumps(vocab))
    (bundle / 'evaluation_index.json').write_text(json.dumps(index))
    train_tokens = _stream(vocab, ['BOS', 'KIND_ENTRY', 'KIND_LINEAR', 'KIND_LINEAR',
                                  'KIND_MERGE', 'REF_1', 'REF_2', 'EOS'])
    write_jsonl(bundle / 'train.jsonl', [{'sample_id': 'train', 'tokens': train_tokens}])
    write_jsonl(bundle / 'val.jsonl', [])
    test = [{'sample_id': s, 'tokens': short if s == 'b' else tokens} for s in ('a', 'b', 'c')]
    write_jsonl(bundle / 'test.jsonl', test)
    for config in ('base', 'mask'):
        for seed in (0, 1):
            path = runs / ce.run_dir_name('b_', config, 24, seed)
            path.mkdir(parents=True)
            rows = scores if config == 'base' else [
                dict(r, nll_per_token=r['nll_per_token'] - (seed+1)*.2,
                     nll=r['nll'] - r['n_tokens']*(seed+1)*.2,
                     token_nll=[v-(seed+1)*.2 for v in r['token_nll']]) for r in reversed(scores)]
            write_jsonl(path / 'test_scores.jsonl', rows)
    args = (runs, 'b_', 24, bundle, ['base', 'mask'], 'base', [0, 1])
    fitted = []
    original_baselines = ce.FrequencyBaselines

    def capture_train(rows, vocab, max_k):
        result = original_baselines(rows, vocab, max_k)
        fitted.append(result.count_k)
        return result

    monkeypatch.setattr(ce, 'FrequencyBaselines', capture_train)
    old = ce.summarize(*args)
    report = ce.summarize(*args, by_bucket=True)
    assert fitted == [{1: 1, 2: 1}, {1: 1, 2: 1}]
    x = report['summary']['mask']['by_bucket']['x']
    assert x['paired_delta_vs_baseline']['mean'] == pytest.approx(-.3)
    assert x['paired_delta_vs_baseline']['sd'] == pytest.approx(.1)
    assert x['nll_per_token']['sd'] == pytest.approx(.1)
    assert report['runs']['mask'][1]['by_bucket']['x']['paired_delta_vs_baseline']['mean_delta'] == pytest.approx(-.4)
    assert report['summary']['mask']['by_bucket']['lost']['paired_delta_vs_baseline']['mean'] is None
    assert 'wf' not in report['runs']['base'][0]['by_bucket']['x']
    # New tables are additive; existing REF frequency baselines and CLI stay unchanged.
    for config in ('base', 'mask'):
        for seed in (0, 1):
            assert {k: v for k, v in report['runs'][config][seed].items() if k != 'by_bucket'} == old['runs'][config][seed]
    fb = original_baselines([{'sample_id': 'train', 'tokens': train_tokens}], vocab, 3)
    assert fb.count_k == {1: 1, 2: 1}
    ce.print_summary(old)
    expected = capsys.readouterr().out
    ce.main(['--runs', str(runs), '--size', '24', '--tokens', str(bundle),
             '--configs', 'base,mask', '--seeds', '0,1'])
    assert capsys.readouterr().out == expected
    ce.print_summary(report)
    assert 'retained 0/1' in capsys.readouterr().out
    # Invalid token and score sets must fail before pairing, including across configs.
    for bad in (test+test[:1], test[:-1], test+[{'sample_id': 'alien', 'tokens': short}]):
        write_jsonl(bundle / 'test.jsonl', bad)
        with pytest.raises(ValueError):
            ce.summarize(*args, by_bucket=True)
    write_jsonl(bundle / 'test.jsonl', test)
    path = runs / 'b_mask_n24_s0' / 'test_scores.jsonl'
    for bad in (scores[:-1], scores+scores[:1], [dict(scores[0], sample_id='alien')]+scores[1:]):
        write_jsonl(path, bad)
        with pytest.raises(ValueError):
            ce.summarize(*args, by_bucket=True)
    path.unlink()
    with pytest.raises(ValueError, match='missing scores'):
        ce.summarize(*args, by_bucket=True)
    # A bundle whose whole test split was excluded still reports null metrics.
    empty_index = dict(index, samples={'train': index['samples']['train']})
    (bundle / 'evaluation_index.json').write_text(json.dumps(empty_index))
    write_jsonl(bundle / 'test.jsonl', [])
    for score_path in runs.glob('*/test_scores.jsonl'):
        write_jsonl(score_path, [])
    write_jsonl(path, [])
    empty_report = ce.summarize(*args, by_bucket=True)
    assert empty_report['summary']['base']['test_nll_per_token']['mean'] is None
    assert empty_report['runs']['mask'][0]['by_bucket']['lost']['n_samples'] == 0
    json.dumps(empty_report, allow_nan=False)
    ce.print_summary(empty_report)
    capsys.readouterr()
    (bundle / 'evaluation_index.json').write_text('{"version":1,"samples":{"a":{},"a":{}}}')
    with pytest.raises(ValueError, match='duplicate'):
        ce.summarize(*args, by_bucket=True)
    (bundle / 'evaluation_index.json').unlink()
    with pytest.raises(FileNotFoundError):
        ce.summarize(*args, by_bucket=True)
