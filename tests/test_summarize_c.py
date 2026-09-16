"""C 集計の完全性と表示を合成データで検証する。"""
import json

import pytest

from cfg_reducer.model_input import build_vocab
from training import summarize_c as sc
from training.controlled_eval import run_dir_name
from training.data_utils import write_jsonl


@pytest.fixture
def cell_runs(tmp_path):
    bundle = tmp_path / 'tokens'
    bundle.mkdir()
    vocab = build_vocab(1)
    (bundle / 'vocab.json').write_text(json.dumps(vocab))
    (bundle / 'meta.json').write_text(json.dumps({
        'counts': {'test': {'raw': 4, 'retained': 2, 'excluded': 2}},
        'exclusions': [{'split': 'test', 'reason': reason}
                       for reason in ('over_window', 'over_length')]}))
    tokens = [vocab[t] for t in ('BOS', 'KIND_ENTRY', 'EOS')]
    write_jsonl(bundle / 'train.jsonl', [{'sample_id': 'train', 'tokens': tokens}])
    write_jsonl(bundle / 'test.jsonl', [{'sample_id': sid, 'tokens': tokens} for sid in ('a', 'b')])
    runs = tmp_path / 'runs'
    for cfg, value in [('base', 1.), ('mask', .5)]:
        run = runs / run_dir_name('test_', cfg, 24, 0)
        run.mkdir(parents=True)
        write_jsonl(run / 'test_scores.jsonl', [dict(sample_id=sid, n_tokens=2,
            nll=2*value, nll_per_token=value, token_nll=[value]*2) for sid in ('a', 'b')])
    cell = dict(key='layered_lay2str', label='test OOD', prefix='test_',
                tokens=str(bundle), configs=['base', 'mask'])
    return cell, runs


def test_strict_missing_seeds_have_no_table(cell_runs):
    cell, runs = cell_runs
    result = sc.collect_cell(cell, runs, 24, [0, 1])
    assert result['status'] == 'pending' and result['incomplete']
    assert len(result['missing']) == 2
    text = '\n'.join(sc.cell_table(cell, result))
    assert 'incomplete:' in text and '| config |' not in text
    assert 'retained 2 / raw 4' in text
    assert 'over_window 1, over_length 1' in text


def test_allow_partial_cli_has_counts(cell_runs, tmp_path, monkeypatch):
    cell, runs = cell_runs
    monkeypatch.setitem(sc.CELL_BY_KEY, cell['key'], cell)
    out = tmp_path / 'summary.md'
    args = ['--runs', str(runs), '--cells', cell['key'], '--seeds', '0,1', '--out', str(out)]
    sc.main(args)
    assert '\n| config |' not in out.read_text()
    sc.main([*args, '--allow-partial'])
    result = json.loads(out.with_suffix('.json').read_text())[cell['key']]
    assert result['status'] == 'partial'
    assert result['summary']['mask']['n_seeds'] == 1
    assert result['summary']['mask']['paired_n_samples'] == {'0': 2}
    assert '| mask | 1 / 2 | 0:2 |' in out.read_text()


def test_pending_score_file_and_validation_error_are_distinct(cell_runs):
    cell, runs = cell_runs
    path = runs / run_dir_name('test_', 'mask', 24, 0) / 'test_scores.jsonl'
    path.write_text('{bad json\n')
    report = sc.collect_cell(cell, runs, 24, [0])
    assert report['status'] == 'error'
    assert '_error_' in '\n'.join(sc.cell_table(cell, report))
    path.unlink()
    report = sc.collect_cell(cell, runs, 24, [0])
    assert report['status'] == 'pending'
    assert report['missing'] == [str(path)]


def test_target_table_best_source_ties_and_missing():
    def report(value):
        return {'n_test': 2, 'summary': {'mask': {'seeds': [0, 1],
            'test_nll_per_token': {'mean': value, 'sd': .1}}}}
    results = {'layered_id': report(2.), 'structured_str2lay': report(1.),
               'mixed_mix2lay': report(1.), 'layered_lay2str': {'status': 'error'}}
    text = '\n'.join(sc.target_table(results))
    line = next(line for line in text.splitlines() if line.startswith('| layered |'))
    assert line.count('**1.0000 ± 0.1000 (2 / 2)**') == 2
    assert '**2.0000' not in line
    assert '| — | — |' in line
    assert '| structured | — | — | — | — | — |' in text


def test_extended_cells_and_ood_counts(cell_runs):
    assert len(sc.CELLS) == 21 and len(sc.OOD_PAIRS) == 15
    cell, runs = cell_runs
    report = sc.collect_cell(cell, runs, 24, [0])
    assert report['status'] == 'complete'
    results = {'layered_id': report, 'layered_lay2str': report}
    text = '\n'.join(sc.ood_table(results, set(results)))
    assert '| mask | 1 / 2 | 1 / 2 |' in text
    (runs.parent / 'tokens' / 'meta.json').write_text('{}')
    report = sc.collect_cell(cell, runs, 24, [0])
    assert 'n/a' in '\n'.join(sc.cell_table(cell, report))
