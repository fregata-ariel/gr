"""Plan generation and actual CLI dry runs, without either compute backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.pretrain.make_plan import main, make_plan
from training.eval_buckets import ALL_NS, read_json
from training.runner.types import plan_from_json, plan_to_json, validate_plan


def test_six_runs_sixty_six_rescores():
    plans = [make_plan('compare')[0], make_plan('final', 'alibi')[0]]
    assert sum(len(p.blocks[0].train) for p in plans) == 6
    assert sum(len(p.blocks[0].rescore) for p in plans) == 66
    names = []
    for phase, plan in zip(('compare', 'final'), plans):
        validate_plan(plan)
        assert plan_from_json(plan_to_json(plan)) == plan
        assert len(plan.blocks) == 1
        block = plan.blocks[0]
        assert block.source_bundle == 'data/tok_pretrain_n48'
        assert len(block.train) == 3
        for job in block.train:
            names.append(job.name)
            assert job.epochs == 60 and job.patience == 10
            assert job.sample_seed == 1000 + job.seed
            assert job.num_samples == job.constrained_samples == (100 if phase == 'compare' else 200)
            assert job.extra[job.extra.index('--wf-probes') + 1] == ('0' if phase == 'compare' else '5')
            for flag in ('--ref-legal-mask', '--ref-diagnostics', '--length-buckets'):
                assert flag in job.extra
            assert {j.out_run for j in block.rescore if j.source_run == job.name} == {f'{job.name}__n{n}' for n in ALL_NS}
        _, index = make_plan(phase, 'alibi')
        assert len(index['runs']) == 3
        assert len(index['buckets']) == 11
        for job, run in zip(block.train, index['runs']):
            assert run['name'] == job.name
            assert run['source_run'] == f'runs/{job.name}'
            assert run['scores_by_n'] == {str(n): f'runs/{job.name}__n{n}/test_scores.jsonl' for n in ALL_NS}
    assert len(set(names)) == 6
    assert [j.seed for j in plans[0].blocks[0].train] == [0, 0, 0]
    assert [j.seed for j in plans[1].blocks[0].train] == [0, 1, 2]


def test_requires_bundles_and_explicit_final_pos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        main(['--phase', 'compare', '--out-dir', 'plans'])
    assert not (tmp_path / 'plans').exists()
    with pytest.raises(SystemExit):
        main(['--phase', 'final', '--out-dir', 'plans'])
    with pytest.warns(UserWarning, match='Missing bundles'):
        main(['--phase', 'compare', '--out-dir', 'plans', '--allow-missing-bundles'])
    assert read_json(tmp_path / 'plans/eval_index_compare.json')['schema_version'] == 1


@pytest.mark.parametrize('backend', ['colab', 'local'])
@pytest.mark.parametrize('phase', ['compare', 'final'])
def test_runner_cli_dry_run_both_backends(tmp_path, backend, phase):
    repo = Path(__file__).resolve().parents[1]
    generated = tmp_path / 'plans'
    env = {**os.environ, 'PYTHONPATH': str(repo)}
    result = subprocess.run([sys.executable, str(repo / 'experiments/pretrain/make_plan.py'),
                             '--phase', phase, '--pos', 'none', '--out-dir', str(generated), '--allow-missing-bundles'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = plan_from_json((generated / f'plan_{phase}.json').read_text())
    training = tmp_path / 'training'
    training.mkdir()
    for filename in ('train_ar.py', 'grammar_mask.py'):
        (training / filename).write_text('')
    for block in plan.blocks:
        bundle = tmp_path / block.source_bundle
        bundle.mkdir(parents=True)
        for filename in ('train.jsonl', 'val.jsonl', 'test.jsonl', 'vocab.json', 'meta.json'):
            (bundle / filename).write_text('')
        for job in block.rescore:
            target = tmp_path / job.target_bundle
            target.mkdir(parents=True, exist_ok=True)
            (target / 'test.jsonl').touch()
    dry = subprocess.run([sys.executable, '-m', 'training.runner', 'run',
                          '--plan', str(generated / f'plan_{phase}.json'), '--repo-root', str(tmp_path),
                          '--runs-dir', str(tmp_path / 'runs'), '--backend', backend, '--dry-run'],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert 'RUN-DONE' in dry.stdout
    for job in plan.blocks[0].train:
        assert job.name in dry.stdout
    for job in plan.blocks[0].rescore:
        assert job.out_run in dry.stdout
    assert not (tmp_path / 'runs').exists()
