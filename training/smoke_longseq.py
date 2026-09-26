"""長系列の独立受入試験。torch のある環境でのみ実行する。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback
from unittest.mock import patch

import torch  # ty: ignore[unresolved-import]
import torch.nn.functional as F  # ty: ignore[unresolved-import]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same_state(a, b, label):
    require(list(a) == list(b), f"{label}: state_dict key/order mismatch")
    for key in a:
        require(torch.equal(a[key], b[key]), f"{label}: tensor mismatch: {key}")


def bundle(root):
    names = ['PAD', 'BOS', 'EOS', 'KIND_ENTRY', 'KIND_LINEAR', 'KIND_MERGE',
             'KIND_LOOP', 'LOOP_START', 'LOOP_END', 'REF_1']
    vocab = {name: i for i, name in enumerate(names)}
    rows: list[dict] = [{'sample_id': f'tiny-{i}', 'seed': i,
             'tokens': [1, 3] + [4, 9] * (i % 3 + 1) + [2]} for i in range(12)]
    meta = {'max_len': max(len(r['tokens']) for r in rows)}
    for name, data in [('vocab', vocab), ('meta', meta)]:
        (root / f'{name}.json').write_text(json.dumps(data))
    for split, data in [('train', rows[:8]), ('val', rows[8:10]), ('test', rows[10:])]:
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in data))
    return vocab, meta, rows


def arguments(module, root, extra=()):
    argv = []
    for name in ('train', 'val', 'test'):
        argv += [f'--{name}', str(root / f'{name}.jsonl')]
    for name in ('vocab', 'meta'):
        argv += [f'--{name}', str(root / f'{name}.json')]
    argv += ['--out', str(root / 'run'), '--epochs', '1', '--batch-size', '3',
             '--d-model', '16', '--nhead', '4', '--num-layers', '1',
             '--dim-feedforward', '24', '--num-samples', '2']
    argv += list(extra)
    return module.build_parser().parse_known_args(argv)[0], argv


def legacy_check(old, new, root, vocab, meta, rows, device):
    results = []
    for module in (old, new):
        args, _ = arguments(module, root)
        vc = module.Vocab(vocab)
        torch.manual_seed(17)
        model = module.build_model(vc, meta, args, 2 * meta['max_len']).to(device)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        initial_rng = torch.get_rng_state().clone()
        seqs = [r['tokens'] for r in rows[:8]]
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(17)
        metrics = module.run_epoch(model, seqs, [{} for _ in seqs], 3, vc, device,
                                   optimizer, rng)
        scores = module.score_rows(model, rows[8:], vc, device, False, False)
        samples = [module.sample_stream(model, vc, device, 18, 1., 0) for _ in range(2)]
        results.append((initial, initial_rng, model.state_dict(), metrics, scores,
                        samples, torch.get_rng_state().clone(), rng.getstate()))
    a, b = results
    same_state(a[0], b[0], 'legacy initialization')
    require(torch.equal(a[1], b[1]), 'legacy initialization RNG differs')
    same_state(a[2], b[2], 'legacy epoch updates')
    require(a[3:6] == b[3:6], 'legacy metrics/score_rows/sample token IDs differ')
    require(torch.equal(a[6], b[6]) and a[7] == b[7], 'legacy final RNG differs')
    require([r['sample_id'] for r in b[4]] == [r['sample_id'] for r in rows[8:]],
            'legacy score sample IDs/order differ')


def attention_check(module, device):
    require(module.alibi_slopes(3) == [1/16, 1/256, 1/4], 'H=3 slope order incorrect')
    require(module.alibi_slopes(4) == [1/4, 1/16, 1/64, 1/256], 'H=4 slopes incorrect')
    for padded in (False, True):
        for training in (True, False):
            label = f'ALiBi padding={padded} training={training}'
            torch.manual_seed(3)
            model = module.ARBaseline(12, 8, 0, d_model=16, nhead=4, num_layers=1,
                                      dim_feedforward=24, dropout=0, pos='alibi').to(device)
            model.train(training)
            x = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]], device=device)
            if padded:
                x[0, -2:] = 0
                x[1, -1:] = 0
            layer = model.encoder.layers[0]
            captured = []
            handle = layer.self_attn.register_forward_hook(lambda m, a, out: captured.append(out[0]))
            previous = torch.backends.mha.get_fastpath_enabled()
            with torch.set_grad_enabled(training):
                actual = model.encode(x)
                handle.remove()
                require(torch.backends.mha.get_fastpath_enabled() == previous,
                        label + ': fastpath not restored')
                h = layer.norm1(model.tok_emb(x))
                q, k, v = F.linear(h, layer.self_attn.in_proj_weight,
                                   layer.self_attn.in_proj_bias).chunk(3, -1)
                q, k, v = [t.reshape(2, 7, 4, 4).transpose(1, 2) for t in (q, k, v)]
                i = torch.arange(7, device=device)
                distance = i[:, None] - i[None, :]
                slopes = torch.tensor([1/4, 1/16, 1/64, 1/256], device=device)
                logits = q @ k.transpose(-1, -2) / math.sqrt(4)
                logits = logits - slopes[None, :, None, None] * distance
                logits = logits.masked_fill(distance[None, None] < 0, float('-inf'))
                logits = logits.masked_fill(x.eq(0)[:, None, None, :], float('-inf'))
                expected = (logits.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 16)
                expected = layer.self_attn.out_proj(expected)
                torch.testing.assert_close(captured[0], expected, rtol=2e-5, atol=2e-6,
                                           msg=label + ': attention hand computation')
                torch.testing.assert_close(actual, model.tok_emb(x) + expected + layer._ff_block(
                    layer.norm2(model.tok_emb(x) + expected)), rtol=2e-5, atol=2e-6,
                    msg=label + ': encoder ignored attention bias')
                require(bool(torch.isfinite(actual).all()), label + ': nonfinite output')
                if training:
                    actual.square().sum().backward()
                    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for p in model.encoder.parameters()), label + ': invalid gradients')
            with torch.no_grad():
                torch.testing.assert_close(model.encode(x), actual, rtol=2e-5, atol=2e-6,
                                           msg=label + ': unhooked encoder differs')
                mask, padding = model.attention_masks(x, model.tok_emb(x))
                require(mask.shape == (8, 7, 7) and padding.shape == (2, 7),
                        label + ': incorrect mask shapes')
                require(mask.dtype == padding.dtype == model.tok_emb.weight.dtype,
                        label + ': masks must be float hidden dtype')
                changed = x.clone()
                changed[:, 4] = 10
                torch.testing.assert_close(model.encode(changed)[:, :4], actual[:, :4],
                                           rtol=0, atol=0, msg=label + ': future leakage')
                model.alibi_slopes.zero_()
                require(not torch.allclose(model.encode(x), actual), label + ': slopes have no effect')
    # Exercise finally even when the encoder fails.
    def fail(*args, **kwargs):
        raise RuntimeError('injected encoder failure')
    handle = model.encoder.register_forward_pre_hook(fail)
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        model.encode(x)
    except RuntimeError as error:
        require(str(error) == 'injected encoder failure', 'unexpected encoder failure')
    finally:
        handle.remove()
    require(torch.backends.mha.get_fastpath_enabled() == previous, 'fastpath not restored on failure')


def bucket_check(module, device):
    seqs = [[i + 1] * n for i, n in enumerate([9, 3, 5, 3, 8, 2, 7, 4, 6])]
    aux = [{'depth': [i + 1] * len(s)} for i, s in enumerate(seqs)]
    def epochs():
        rng = random.Random(5)
        result = []
        for _ in range(2):
            batches = list(module.iter_batches(seqs, aux, 3, 0, device, rng,
                                              length_buckets=True, bucket_size=4))
            ids = [b['tokens'][:, 0].tolist() for b in batches]
            require(sorted(i for batch in ids for i in batch) == list(range(1, 10)),
                    'buckets lost/duplicated samples')
            require(sorted(map(len, ids)) == [1, 1, 1, 3, 3], 'bucket tails merged/dropped')
            for b in batches:
                require(torch.equal(b['tokens'], b['depth']), 'bucket aux alignment lost')
            result.append(ids)
        return result
    before = torch.get_rng_state().clone()
    result = epochs()
    require(result == epochs(), 'bucket epoch history not reproducible')
    require(result[0] != result[1], 'bucket RNG not carried across epochs')
    require(torch.equal(before, torch.get_rng_state()), 'bucketing consumed torch RNG')
    batches = module.iter_batches(seqs, aux, 3, 0, device, length_buckets=True, bucket_size=4)
    require([i for b in batches for i in b['tokens'][:, 0].tolist()] ==
            [i + 1 for i in sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))],
            'validation bucket stable ordering differs')


def modes_check(module, root, vocab, meta, rows, device):
    vc = module.Vocab(vocab)
    for pos in ('learned', 'sinusoidal', 'alibi', 'none'):
        args, argv = arguments(module, root, ['--pos', pos, '--max-len', '7',
                                             '--length-buckets', '--bucket-size', '4'])
        with patch.object(torch.cuda, "is_available", return_value=False):
            module.main(argv)
        stats = json.loads((root / 'run/length_stats.json').read_text())
        require(stats['kept']['samples'] == 6 and stats['dropped']['samples'] == 2,
                pos + ': incorrect max-len counts')
        history = json.loads((root / 'run/history.json').read_text())[0]
        require(history['train_tokens'] == 30 and history['train_kept'] == 6 and
                history['train_dropped_over_max_len'] == 2, pos + ': incorrect history')
        config = json.loads((root / 'run/samples.json').read_text())['config']
        if pos != 'learned':
            require(config['gen_max_len'] == 14, pos + ': generation budget not resolved')
        for split in ('val', 'test'):
            scores = module.load_rows(root / f'run/{split}_scores.jsonl')
            require(len(scores) == 2, pos + ': evaluation was filtered')
        model = module.build_model(vc, meta, args, 18).to(device)
        model.load_state_dict(torch.load(root / 'run/model.pt', map_location=device))
        long_row = {'sample_id': 'long', 'tokens': [1, 3] + [4, 9] * 6 + [2]}
        score = module.score_rows(model, [long_row], vc, device, False, False)[0]
        require(score['n_tokens'] == 14 and math.isfinite(score['nll']), pos + ': long scoring failed')
        if pos == 'learned':
            try:
                model.encode(torch.ones((1, 19), dtype=torch.long, device=device))
            except ValueError as error:
                require('capacity' in str(error), 'learned overflow error unclear')
            else:
                raise AssertionError('learned overflow accepted')
        if pos == 'sinusoidal':
            keys = list(model.state_dict())
            # Score beyond 4096, using a narrow model to bound CPU work.
            tiny = module.ARBaseline(len(vocab), 18, 0, d_model=4, nhead=1,
                                     num_layers=1, dim_feedforward=4, dropout=0,
                                     pos='sinusoidal').to(device)
            rng = torch.get_rng_state().clone()
            row = {'tokens': [1, 3] + [4, 9] * 2048 + [2]}
            score = module.score_rows(tiny, [row], vc, device, False, False)[0]
            require(score['n_tokens'] == 4098, 'sinusoidal >4096 scoring truncated')
            require(tiny.pos_table.size(0) == 8192, 'sinusoidal growth rule incorrect')
            require(torch.equal(rng, torch.get_rng_state()), 'sinusoidal growth consumed RNG')
            require('pos_table' not in tiny.state_dict() and keys == list(model.state_dict()),
                    'sinusoidal buffer persisted')


def structural_check(module, device):
    for mode in ('learned', 'sinusoidal', 'depth_only'):
        model = module.ARBaseline(12, 4, 0, d_model=6, nhead=2, num_layers=1,
                                  dropout=0, use_struct=True, struct_mode=mode,
                                  pos='sinusoidal', pos_table_len=2).to(device)
        x = torch.ones((1, 7), dtype=torch.long, device=device)
        depth = torch.zeros_like(x)
        lpos = torch.arange(7, device=device)[None]
        model.eval()
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            output = model.encode(x, depth, lpos)
        require(bool(torch.isfinite(output).all()), mode + ': structural output nonfinite')
        require(torch.equal(rng, torch.get_rng_state()), mode + ': structural growth consumed RNG')
        if mode == 'sinusoidal':
            require(model.lpos_table.size(0) == 7 and 'lpos_table' not in model.state_dict(),
                    'structural sinusoidal table did not grow independently')
        elif mode == 'learned':
            with torch.no_grad():
                torch.testing.assert_close(output, model.encode(x, depth, lpos.clamp(max=3)))
    try:
        module.ARBaseline(12, 8, 0, pos='none', use_struct=True)
    except ValueError as error:
        require('--struct-pos' in str(error), 'none/struct rejection unclear')
    else:
        raise AssertionError('none accepted structural positions')
    table = module.sinusoidal_table(7, 5)
    require(table.shape == (7, 5) and bool(torch.isfinite(table).all()),
            'odd-width sinusoidal table failed')


def operation_check(module, root, vocab, meta, rows, device):
    vc = module.Vocab(vocab)
    source = root / 'operation_source'
    _, argv = arguments(module, root, ['--out', str(source), '--pos', 'sinusoidal',
                                      '--ref-diagnostics', '--constrained-samples', '2'])
    with patch.object(torch.cuda, 'is_available', return_value=False):
        module.main(argv)
    config = json.loads((source / 'config.json').read_text())
    require(config['schema_version'] == 1 and config['vocab'] == vocab, 'config contract')
    require(config['vocab_sha256'] == module.vocab_hash(vocab), 'config vocab hash')
    require(config['args']['model_max_len'] == config['model']['max_len'], 'resolved capacity')
    args = argparse.Namespace(**config['args'])
    model = module.build_model(vc, meta, args, args.gen_max_len).to(device)
    module.initialize_weights(model, config, source, device)
    same_state(model.state_dict(), torch.load(source / 'model.pt', weights_only=True,
                                             map_location=device), 'init weights')
    # Different target metadata must not resize the saved model.
    rebuilt = module.build_model(vc, {'max_len': 100}, args, 200).to(device)
    require(rebuilt.max_len == model.max_len, 'rescore changed model capacity')
    target = root / 'operation_target'
    with patch.object(torch.cuda, 'is_available', return_value=False):
        module.main(argv + ['--out', str(target), '--init-from', str(source), '--seed', '19'])
    initialized = json.loads((target / 'config.json').read_text())
    require(initialized['init_from']['model.pt_sha256'], 'missing source hash')
    require(initialized['args']['seed'] == 19, 'init did not use fresh seed')
    require(json.loads((target / 'history.json').read_text())[0]['epoch'] == 1, 'init resumed epoch')

    def rejects(call, message):
        try:
            call()
        except ValueError as error:
            require(message in str(error), f'unclear rejection: {error}')
        else:
            raise AssertionError('accepted mismatch: ' + message)

    for flags, message in [(['--pos', 'alibi'], 'pos'),
                           (['--d-model', '20'], 'd_model'),
                           (['--nhead', '2'], 'nhead'),
                           (['--out', str(source)], '--out')]:
        with patch.object(torch.cuda, 'is_available', return_value=False):
            rejects(lambda: module.main(argv + ['--out', str(target), '--init-from', str(source)]
                                        + flags), message)
    swapped = dict(vocab)
    swapped['KIND_ENTRY'], swapped['KIND_LINEAR'] = swapped['KIND_LINEAR'], swapped['KIND_ENTRY']
    changed = root / 'swapped.json'
    changed.write_text(json.dumps(swapped))
    rejects(lambda: module.main(argv + ['--out', str(target), '--init-from', str(source),
                                       '--vocab', str(changed)]), 'vocab')
    old_run = root / 'old_run'
    old_run.mkdir()
    rejects(lambda: module.main(argv + ['--out', str(target), '--init-from', str(old_run)]), 'config.json')
    state = torch.load(source / 'model.pt', weights_only=True, map_location=device)
    original = state['head.weight']
    for mutation in ('shape', 'missing', 'extra'):
        broken = dict(state)
        if mutation == 'shape':
            broken['head.weight'] = original[:-1]
        elif mutation == 'missing':
            del broken['head.weight']
        else:
            broken['unexpected'] = original
        torch.save(broken, target / 'model.pt')
        rejects(lambda: module.initialize_weights(model, config, target, device), 'state_dict')

    for key in ('dropout', 'struct_mode', 'pointer_legal', 'max_k', 'ref_legal_mask',
                'table_capacities', 'pos_table_len'):
        incomplete = {**config, 'model': dict(config['model'])}
        del incomplete['model'][key]
        rejects(lambda: module.initialize_weights(model, incomplete, source, device), key)
    # Allocation size of nonpersistent sinusoidal tables may differ.
    resized = argparse.Namespace(**vars(args))
    resized.pos_table_len = 7
    module.initialize_weights(module.build_model(vc, meta, resized, 18).to(device),
                              config, source, device)
    for flags in ([], ['--ref-legal-mask'], ['--pointer'], ['--pointer', '--pointer-legal']):
        diag_args, _ = arguments(module, root, ['--pos', 'sinusoidal', '--ref-diagnostics'] + flags)
        diagnostic = module.build_model(vc, meta, diag_args, 18).to(device)
        scores = module.score_rows(diagnostic, rows[:3], vc, device, False, bool(flags))
        for row, score in zip(rows, scores):
            aux = module.sequence_aux(row['tokens'], vc, False, True)
            require(len(score['ref_pred_k']) == len(score['ref_pos']), 'REF diagnostic alignment')
            require(score['ref_pred_legal'] == [aux['klast'][t] < k <= aux['plpos'][t] - 1
                    for t, k in zip(score['ref_pos'], score['ref_pred_k'])], 'REF legality')
        if flags:
            require(all(all(s['ref_pred_legal']) for s in scores), 'legal REF prediction invalid')
    bad = module.build_model(vc, meta, diag_args, 18).to(device)
    rejects(lambda: module.score_rows(bad, [{'tokens': [1, 9, 2]}], vc, device, False, True),
            'data contract')

    # Direct scoring also checks isolation before ordinary sampling resets its seed.
    for count in (0, 2):
        model.wf_probes = count
        torch.manual_seed(47)
        before = torch.get_rng_state().clone()
        scores = module.score_rows(model, rows[8:][::-1], vc, device, False, False)
        require(torch.equal(before, torch.get_rng_state()), 'probe changed CPU RNG')
        samples = [module.sample_stream(model, vc, device, 18, 1., 0) for _ in range(2)]
        if count == 0:
            expected_scores, expected_samples = scores, samples
        else:
            require(samples == expected_samples, 'probes changed samples')
            require([{k: v for k, v in s.items() if k != 'wf_probe'} for s in scores]
                    == expected_scores, 'probes changed scores')
            selected = sorted(r['sample_id'] for r in rows[8:])[:2]
            for row, score in zip(rows[8:][::-1], scores):
                require(('wf_probe' in score) == (row['sample_id'] in selected), 'probe ID order')
                if 'wf_probe' in score:
                    probe = score['wf_probe']
                    prefix = row['tokens'][:max(1, (len(row['tokens']) - 1) // 2)]
                    expected_seed = int.from_bytes(module.hashlib.sha256(
                        f"{args.seed}:{row['sample_id']}:wf-v1".encode()).digest()[:8], 'big')
                    require(probe['seed'] == expected_seed, 'probe seed derivation')
                    require(probe['prefix_len'] == len(prefix) and probe['budget'] == 2 * len(row['tokens']),
                            'probe prefix/budget')
                    for key in ('raw', 'constrained'):
                        require(probe[key][:len(prefix)] == prefix and len(probe[key]) <= probe['budget'],
                                'probe continuation contract')
                    state_machine = module.grammar_mask.GrammarState(vocab)
                    for token in probe['constrained'][1:]:
                        require(token in state_machine.allowed_ids(), 'probe grammar violation')
                        state_machine.push(token)
    probe_run = root / 'operation_probes'
    with patch.object(torch.cuda, 'is_available', return_value=False):
        module.main(argv + ['--out', str(probe_run), '--wf-probes', '2'])
    saved_args = argparse.Namespace(**json.loads((probe_run / 'samples.json').read_text())['config'])
    rescored = module.build_model(vc, meta, saved_args, saved_args.gen_max_len).to(device)
    rescored.load_state_dict(torch.load(probe_run / 'model.pt', map_location=device, weights_only=True))
    require(module.score_rows(rescored, rows[10:], vc, device, False, False) ==
            module.load_rows(probe_run / 'test_scores.jsonl'), 'rescoring did not reproduce probes')
    for filename in ('samples.json', 'samples_constrained.json'):
        require(json.loads((source / filename).read_text())['samples'] ==
                json.loads((probe_run / filename).read_text())['samples'], 'probe run changed samples')
    for split in ('val', 'test'):
        actual = module.load_rows(probe_run / f'{split}_scores.jsonl')
        require([{k: v for k, v in s.items() if k != 'wf_probe'} for s in actual] ==
                module.load_rows(source / f'{split}_scores.jsonl'), 'probe run changed scoring')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu'])
    parser.add_argument('--out', type=Path, default=Path('/tmp/smoke_longseq.json'))
    args = parser.parse_args()
    summary = {'torch_version': torch.__version__, 'image_id': os.environ.get('HOSTNAME'),
               'device': args.device, 'checks': [], 'passed': False}
    try:
        torch.set_num_threads(1)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        old = load(args.legacy, 'legacy_train_ar')
        new = load(Path(__file__).with_name('train_ar.py'), 'current_train_ar')
        with tempfile.TemporaryDirectory(prefix='gr-longseq-') as directory:
            root = Path(directory)
            vocab, meta, rows = bundle(root)
            for name, check in [
                ('legacy', lambda: legacy_check(old, new, root, vocab, meta, rows, args.device)),
                ('alibi', lambda: attention_check(new, args.device)),
                ('buckets', lambda: bucket_check(new, args.device)),
                ('structural', lambda: structural_check(new, args.device)),
                ('operation', lambda: operation_check(new, root, vocab, meta, rows, args.device)),
                ('modes', lambda: modes_check(new, root, vocab, meta, rows, args.device)),
            ]:
                check()
                summary['checks'].append(name)
        summary['passed'] = True
    except Exception:
        summary['error'] = traceback.format_exc()
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
