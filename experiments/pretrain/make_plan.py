"""Write the comparison/final pretraining Plan and its evaluation index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

# Support the documented direct script invocation as well as python -m.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.eval_buckets import ALL_NS, check_bundles
from training.runner.types import (Block, Plan, RescoreJob, TrainJob,
                                   plan_from_json, plan_to_json, validate_plan)

POSITIONS = ('sinusoidal', 'alibi', 'none')


def make_plan(phase: str, pos: str | None = None) -> tuple[Plan, dict]:
    if phase not in ('compare', 'final') or (phase == 'final' and pos not in POSITIONS):
        raise ValueError('final phase requires --pos sinusoidal|alibi|none')
    settings = [(p, 0) for p in POSITIONS] if phase == 'compare' else [(pos, s) for s in range(3)]
    jobs, rescores, runs = [], [], []
    for position, seed in settings:
        name = f"pretrain_{'cmp' if phase == 'compare' else 'final'}_{position}_s{seed}"
        samples, probes = (100, 0) if phase == 'compare' else (200, 5)
        extra = ('--pos', str(position), '--pos-table-len', '4096', '--ref-legal-mask',
                 '--length-buckets', '--max-len', '0', '--ref-diagnostics', '--wf-probes', str(probes),
                 '--d-model', '128', '--num-layers', '4', '--nhead', '4', '--dim-feedforward', '512')
        jobs.append(TrainJob(name, 60, 10, samples, samples, seed, 1000 + seed, extra))
        paths = {}
        for n in ALL_NS:
            out_run = f'{name}__n{n}'
            rescores.append(RescoreJob(name, f'data/tok_pretrain_n{n}', out_run))
            paths[str(n)] = f'runs/{out_run}/test_scores.jsonl'
        runs.append(dict(name=name, pos=position, seed=seed, source_run=f'runs/{name}', scores_by_n=paths))
    source = 'data/tok_pretrain_n48'
    plan = Plan(f'pretrain_{phase}_v1', (Block(source, tuple(jobs), tuple(rescores)),))
    validate_plan(plan)
    if plan_from_json(plan_to_json(plan)) != plan:
        raise ValueError('Plan JSON round-trip failed')
    index = {'schema_version': 1, 'source_dataset': 'data/pretrain_mix', 'source_bundle': source,
             'buckets': [dict(n=n, dataset=f'data/pretrain_test_n{n}', bundle=f'data/tok_pretrain_n{n}') for n in ALL_NS],
             'runs': runs, 'selection_protocol': {
                 'phase': phase, 'selection_n': 192, 'held_out_n': 256,
                 'primary': 'n192 retained token-micro NLL minimum',
                 'equivalence': 'paired token-weighted bootstrap CI95 includes zero',
                 'tie_break_order': ['teacher-forced REF violations', 'prefix-continuation WF', 'seconds/epoch'],
                 'n256_policy': '選択固定後に開く。方式選択には使わない。',
                 'comparison_prefix_wf': '比較phaseはprobes=0のためWFによる同点判定は欠測。',
                 'selection_record': 'selection.jsonにchosen_posとID9 bucketの代償を記録する。',
                 'small_bucket_policy': 'n8/12はraw<100ならID macroから除外し注記する。',
                 'implementation_checks': 'NaN・異常値・制約生成の文法違反は選択前に検査する。'}}
    return plan, index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('compare', 'final'), required=True)
    parser.add_argument('--pos', choices=POSITIONS)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--allow-missing-bundles', action='store_true')
    args = parser.parse_args(argv)
    if args.phase == 'final' and args.pos is None:
        parser.error('final phase requires --pos')
    plan, index = make_plan(args.phase, args.pos)
    try:
        check_bundles(index, Path('.'))
    except FileNotFoundError:
        if not args.allow_missing_bundles:
            raise
        warnings.warn('Missing bundles: skipped bundle completeness/vocab/source checks.', stacklevel=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f'plan_{args.phase}.json').write_text(plan_to_json(plan), encoding='utf-8')
    (args.out_dir / f'eval_index_{args.phase}.json').write_text(
        json.dumps(index, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
