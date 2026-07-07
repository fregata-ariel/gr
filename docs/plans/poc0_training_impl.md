# PoC-0 Training Pipeline Implementation Plan

This plan is bound by `docs/plans/poc0_training_spec.md`. It records the
implementation choices for the PoC-0 training pipeline before coding. The
implementation phase must not reinterpret the fixed stack, platform split,
tokenization rules, model size band, Colab T4 budget, or eval criteria from
the spec.

## Corpus Probe And Fixed Constants

Probe command run during planning:

```sh
./.venv/bin/python scripts/gen_corpus.py \
  --out /tmp/poc0_plan_probe_rerun.jsonl \
  --min-nodes 6 --max-nodes 20 \
  --edge-probs 0.10,0.15,0.20,0.25,0.30 \
  --seeds-per-config 1000
```

Fresh probe stderr:

```text
generated=75000 discarded=2427 deduped=51883 written=20690
```

Measured from `/tmp/poc0_plan_probe_rerun.jsonl`:

| Value | Measurement |
|---|---:|
| Record count | 20,690 |
| Min raw token length, no BOS | 16 |
| Mean raw token length, no BOS | 40.745 |
| Median raw token length, no BOS | 42 |
| P90 / P95 / P99 raw token length | 54 / 57 / 61 |
| Max raw token length, no BOS | 65 |
| Max length with BOS | 66 |
| Max pointer index observed | 17 |
| Observed pointer tokens | `ptr_0` through `ptr_17` |

Training constants:

| Constant | Value | Rationale |
|---|---:|---|
| `MAX_SEQ_LEN` | 80 | Spec guidance; measured max is 65 raw tokens, 66 with BOS, leaving 14 total-position slots of margin. |
| `PTR_COUNT` | 32 | Observed max pointer index is 17, so 18 pointer ids are required; 32 keeps 14 extra pointer ids and supports `ptr_31`. |
| `VOCAB_SIZE` | 41 | 9 fixed non-pointer tokens plus 32 pointer tokens. |
| Split seed | `20260707` | Stable seeded random split by record. |
| Train/val split rule | `int(0.95 * n_records)` train records after seeded shuffle | For the measured probe this is 19,655 train and 1,035 val records. |

The first probe run in this planning pass reported `written=20686`, while the
fresh rerun reported `written=20690`. The max pointer index and max sequence
length were stable. The spec only fixes the corpus scale as approximately
20.7k unique records, so implementation tests must not assert an exact
corpus count. The implementation must sort or seeded-shuffle loaded records
before splitting so train/val selection is deterministic for any generated
JSONL file.

## Spec Conflicts

No binding spec conflict was found.

Observed caveat: repeated corpus generation produced record counts within the
spec's approximate 20.7k range but not an identical exact count. This does not
change any fixed requirement, but it means the training code must treat the
input JSONL as the source of truth and record its manifest values instead of
assuming a hard-coded count.

## 1. Module Layout

Create a new top-level package `poc0/`. Keep imports safe on macOS: modules
needed by local tests must not import Grain or ArrayRecord at module import
time.

| File | Purpose |
|---|---|
| `poc0/__init__.py` | Package marker and small version/export surface. |
| `poc0/constants.py` | Shared constants: vocab list, `MAX_SEQ_LEN=80`, seeds, default hyperparameters. |
| `poc0/tokenizer.py` | Fixed vocab, token/id conversion, JSONL record encoding, padding and next-token target construction. |
| `poc0/data.py` | Common `Batch`, `DatasetInfo`, `BatchSource` interface, deterministic split logic, and in-memory dataset source. |
| `poc0/array_record_data.py` | Linux-only ArrayRecord conversion and Grain-backed `BatchSource`; imports `grain` and `array_record` only inside guarded functions. |
| `poc0/model.py` | Flax linen decoder-only Transformer and causal mask helpers. |
| `poc0/train.py` | CLI and library entry point for train state creation, train/eval steps, schedule, checkpointing. |
| `poc0/checkpoints.py` | Orbax save/restore helpers with small metadata manifest. |
| `poc0/sample.py` | Autoregressive sampler with depth-0 STOP termination and invalid-length-cap handling. |
| `poc0/stats.py` | `Skeleton`-derived stats equivalent to `scripts/gen_corpus.py::_stats_for`, histogram and TVD helpers. |
| `poc0/eval.py` | CLI and library entry point for sampling N=1000, decode-validity, novelty, distribution metrics, and plots. |
| `poc0/README.md` | Exact Colab workflow created in the implementation phase. |

Planned tests under `tests/`:

| Test file | Coverage |
|---|---|
| `tests/test_poc0_tokenizer.py` | Vocab ids, encode/decode token ids, padding, masks, pointer bounds. |
| `tests/test_poc0_data.py` | Deterministic split and in-memory `BatchSource` contract. |
| `tests/test_poc0_array_record_data.py` | Linux-only ArrayRecord/Grain smoke test, skipped off-Linux or when deps are unavailable. |
| `tests/test_poc0_model.py` | Model init, logits shape, causal mask, parameter count band. |
| `tests/test_poc0_train.py` | Masked loss, schedule values, train step shape/determinism on a tiny batch. |
| `tests/test_poc0_sample.py` | Depth-0 STOP termination, nested STOP handling, length-cap invalid handling. |
| `tests/test_poc0_stats.py` | Skeleton-derived stats and TVD calculations. |
| `tests/test_poc0_eval.py` | Decode-validity, novelty, duplicate-of-train rate, and metric report assembly. |

## 2. Hyperparameters And Budget

Architecture:

| Hyperparameter | Value |
|---|---:|
| Model type | Decoder-only causal Transformer |
| Vocab size | 41 |
| Max sequence length | 80 total positions including BOS |
| Layers | 4 |
| `d_model` | 256 |
| Attention heads | 4 |
| Head dim | 64 |
| MLP hidden dim | 1024 |
| Activation | GELU |
| Normalization | Pre-norm LayerNorm |
| Positional encoding | Learned absolute positional embedding, length 80 |
| Dropout | 0.0 for PoC-0 determinism and simpler local tests |
| LM head | Tied to token embedding |
| Train dtype | float32 |

Parameter-count estimate with Flax Dense biases and tied LM head:

| Component | Count |
|---|---:|
| Token embedding, `41 * 256` | 10,496 |
| Positional embedding, `80 * 256` | 20,480 |
| Attention per layer, `4 * (256 * 256 + 256)` | 263,168 |
| MLP per layer, `256 * 1024 + 1024 + 1024 * 256 + 256` | 525,568 |
| Two LayerNorms per layer, `2 * 2 * 256` | 1,024 |
| One Transformer layer | 789,760 |
| Four Transformer layers | 3,159,040 |
| Final LayerNorm | 512 |
| Total | 3,190,528 |

This lands in the required 2-4M parameter band. An untied head would add only
10,537 parameters and still fit the band, but the implementation should use
the tied head above.

Training hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Adam `b1` / `b2` / `eps` | 0.9 / 0.95 / 1e-8 |
| Weight decay | 0.01 |
| Global grad clip | 1.0 |
| Peak learning rate | 3e-4 |
| Warmup steps | 200 |
| Decay | Cosine decay from 3e-4 to 3e-5 through step 3000 |
| Batch size | 256 |
| Train steps | 3000 |
| Val interval | Every 100 train steps |
| Checkpoints | Orbax every 500 steps, final checkpoint always saved, keep last 3 plus best val |
| Sampling eval | N=1000, temperature 1.0, seed `20260710` |
| Init seed | `20260709` |
| Train shuffle seed | `20260708` |

Budget rationale:

- Measured data has 20,690 records. With the fixed split rule, train has
  19,655 records and val has 1,035 records.
- With batch size 256 and `drop_remainder=True`, one training epoch is 76
  full batches. `3000 / 76 = 39.5` effective epochs over full batches.
- Each step processes `256 * 80 = 20,480` padded positions. The measured mean
  active target count is 40.745 per record, so each step averages about
  10,431 active next-token targets.
- Total padded positions are about 61.4M and active target predictions are
  about 31.3M. A 3.19M parameter, 4-layer, sequence-length-80 model is small
  for a Colab T4; the expected runtime is comfortably below 30 minutes even
  with JAX compile, full validation every 100 steps, and checkpoint writes.

## 3. Vocab Construction

The vocab is fixed and versioned in `poc0/constants.py`. Token ids are stable
and must never be inferred from sorted order.

Full token list and ids:

| Id | Token |
|---:|---|
| 0 | `PAD` |
| 1 | `BOS` |
| 2 | `ADD_ENTRY` |
| 3 | `ADD_LINEAR` |
| 4 | `ADD_MERGE` |
| 5 | `ADD_LOOP` |
| 6 | `OPEN` |
| 7 | `CLOSE` |
| 8 | `STOP` |
| 9 | `ptr_0` |
| 10 | `ptr_1` |
| 11 | `ptr_2` |
| 12 | `ptr_3` |
| 13 | `ptr_4` |
| 14 | `ptr_5` |
| 15 | `ptr_6` |
| 16 | `ptr_7` |
| 17 | `ptr_8` |
| 18 | `ptr_9` |
| 19 | `ptr_10` |
| 20 | `ptr_11` |
| 21 | `ptr_12` |
| 22 | `ptr_13` |
| 23 | `ptr_14` |
| 24 | `ptr_15` |
| 25 | `ptr_16` |
| 26 | `ptr_17` |
| 27 | `ptr_18` |
| 28 | `ptr_19` |
| 29 | `ptr_20` |
| 30 | `ptr_21` |
| 31 | `ptr_22` |
| 32 | `ptr_23` |
| 33 | `ptr_24` |
| 34 | `ptr_25` |
| 35 | `ptr_26` |
| 36 | `ptr_27` |
| 37 | `ptr_28` |
| 38 | `ptr_29` |
| 39 | `ptr_30` |
| 40 | `ptr_31` |

Tokenizer rules:

- Input corpus records contain raw grammar tokens only. They do not contain
  `BOS`, `PAD`, or EOS.
- There is no EOS token. `STOP` is a grammar token and completion is defined
  only by the sampler/evaluator's depth-0 STOP rule.
- For training, construct `full = [BOS] + raw_token_ids`. If `len(full) >
  MAX_SEQ_LEN`, reject the record with an explicit error.
- `input_ids` are `full[:-1]` padded to length 80 with `PAD`.
- `target_ids` are `full[1:]` padded to length 80 with `PAD`.
- `loss_mask` is true for exactly `len(full) - 1` positions, which trains on
  tokens after BOS up to and including the final `STOP`; padding is masked.
- Unknown tokens and pointers outside `ptr_0..ptr_31` are hard errors in data
  conversion and local in-memory loading.

## 4. Dataset Adapter Interface

The Grain/ArrayRecord path and the in-memory substitute implement the same
batch-source contract. The common contract lives in `poc0/data.py`; the
production implementation lives in `poc0/array_record_data.py`.

Shared data objects:

| Object | Fields |
|---|---|
| `Batch` | `input_ids: int32[batch, 80]`, `target_ids: int32[batch, 80]`, `loss_mask: bool[batch, 80]` |
| `DatasetInfo` | `split`, `n_records`, `max_seq_len`, `vocab_size`, `source_path`, `manifest` |

Exact `BatchSource` interface:

```text
info -> DatasetInfo
iter_batches(batch_size, shuffle, seed, drop_remainder, repeat) -> Iterator[Batch]
```

Interface semantics:

- `iter_batches(..., repeat=True)` is infinite and used for training.
- `iter_batches(..., repeat=False)` is finite and used for validation/tests.
- `shuffle=True` means deterministic reshuffling by `seed + epoch_index`.
- `drop_remainder=True` is required for JIT-stable training batches.
- Output arrays are host NumPy arrays with the exact dtypes above. The train
  loop is responsible for `jax.device_put`.
- The interface never exposes Grain objects to model, train, sample, or eval
  code.

In-memory source:

- `InMemoryTokenDataset` reads JSONL, validates tokens, applies the seeded
  split, converts examples to padded arrays, and serves batches from Python
  lists/NumPy arrays.
- It is the default for local macOS tests and for small smoke runs.
- It must produce byte-for-byte identical `Batch` arrays to the production
  source for the same JSONL, vocab, split seed, and batch settings.

Production Grain/ArrayRecord source:

- `convert_jsonl_to_arrayrecord` runs only on Linux. It reads JSONL, applies
  the same tokenizer and split code, writes `train.array_record`,
  `val.array_record`, and a `manifest.json`.
- Each ArrayRecord item stores one already-tokenized example with fixed-shape
  `input_ids`, `target_ids`, `loss_mask`, and `record_id` metadata. Fixed
  shapes keep the Grain map/batch path thin.
- `GrainArrayRecordDataset` reads the manifest and ArrayRecord file, builds a
  Grain source/data-loader, and yields the same `Batch` contract.
- `array_record_data.py` must not import `grain` or `array_record` at module
  import time. It imports them inside Linux-only functions, raises a clear
  `RuntimeError` on non-Linux, and test code skips the smoke test when the
  imports are unavailable.

## 5. Sampling Loop Design

Sampling is intentionally not a full grammar-constrained decoder. The model
must learn the grammar; the sampler only handles special-token masking,
temperature, depth tracking, and termination.

Algorithm:

1. Start with prefix `[BOS]`, empty emitted token list, `depth = 0`, and
   `invalid_reason = None`.
2. For at most `MAX_SEQ_LEN - 1` generated tokens, run the model on the
   current prefix padded to length 80 and take logits at the last real prefix
   position.
3. Set logits for `PAD` and `BOS` to `-inf`; these are never legal emitted
   corpus tokens. Do not mask grammar tokens, pointer tokens, `OPEN`, `CLOSE`,
   or `STOP`.
4. Sample from `softmax(logits / temperature)` with deterministic JAX PRNG
   splitting. PoC-0 eval uses `temperature=1.0`.
5. Append the sampled token to the emitted token list.
6. If token is `OPEN`, increment `depth`.
7. If token is `CLOSE`, decrement `depth`. If depth becomes negative, stop
   immediately and return an invalid sample with reason `negative_depth`.
8. If token is `STOP` and current `depth == 0`, terminate successfully and
   return the emitted tokens, including the final STOP.
9. If the loop reaches the length cap without a depth-0 STOP, return an
   invalid sample with reason `length_cap`.

Important edge cases:

- `STOP` at depth greater than 0 does not terminate the full stream; it is
  allowed to be part of a child level and the sampler continues.
- The sampler does not append a synthetic STOP at length cap. Length cap
  means invalid by definition.
- The evaluator still calls `cfg_reducer.linearize.decode()` on successfully
  terminated samples. Depth tracking is necessary but not sufficient for
  grammatical validity.

## 6. Eval Design

Eval runs locally on CPU against a checkpoint and on Colab after training.
The default command samples `N=1000` sequences at temperature 1.0.

Validity:

- A sample is valid only if the sampler reports successful depth-0 STOP
  termination and `cfg_reducer.linearize.decode(tokens)` accepts the emitted
  raw token list.
- Report `valid_count`, `invalid_count`, validity fraction, and invalid
  reasons: `length_cap`, `negative_depth`, and `decode_error`.
- The target pass bar from the spec is validity >= 99%.

Novelty:

- Build the train set as exact token strings joined with `"\x1f"` from the
  same split manifest used for training.
- For valid samples only, report:
  - novelty rate: valid samples whose exact token string is not in train
  - duplicate-of-train rate: valid samples whose exact token string is in train
  - unique valid sample count and repeated-sample count
- No fixed novelty target is added beyond the spec. A near-zero novelty rate
  is reported as a PoC failure signal.

Skeleton-derived stats:

- Decode valid samples to `cfg_reducer.types.Skeleton`.
- Compute stats from `Skeleton`, mirroring `scripts/gen_corpus.py::_stats_for`:
  - `n_motifs`: recursive count of all skeleton items
  - `kinds`: recursive counts for `entry`, `linear`, `merge`, `loop`
  - `n_tokens`: raw emitted token length
  - `depth`: top-level Kahn-layer depth from item parent indices
  - `max_width`: max top-level Kahn-layer width
  - `loop_nest`: recursive max loop nesting, 0 for no loops
- For train records, use the JSONL `stats` values when available and verify
  on a small subset that `stats_for_skeleton(decode(tokens), tokens)` matches
  the JSONL stats.

Histogram comparison:

- Build normalized histograms for `n_motifs`, `n_tokens`, `depth`,
  `max_width`, `loop_nest`, and per-kind count fields
  `kinds.entry`, `kinds.linear`, `kinds.merge`, `kinds.loop`.
- For each histogram, compute total variation distance:
  `0.5 * sum(abs(p_train(bin) - p_sample(bin)))` over the union of bins.
- Write `metrics.json` with validity, novelty, duplicate rates, per-stat TVD,
  and raw histogram counts.
- Write visual overlays, one PNG per stat, using the existing project
  `matplotlib` dependency.
- The distribution judgment remains qualitative for PoC-0, as required by
  the spec.

## 7. Colab Cell Sequence

All Colab commands assume this repository is available at `/content/gr`.
The binding workflow is: generate JSONL locally on macOS, upload it to Colab,
convert to ArrayRecord on Linux, train, then eval.

Cell 1, install pinned training dependencies:

```sh
%cd /content/gr
!python -m pip install --upgrade pip
!python -m pip install \
  "jax[cuda12]==0.10.2" \
  "flax==0.12.7" \
  "optax==0.2.8" \
  "orbax-checkpoint==0.12.1" \
  "grain==0.2.18" \
  "array-record==0.8.3"
```

Cell 2, upload the locally generated JSONL:

```python
from google.colab import files
uploaded = files.upload()  # upload poc0_corpus.jsonl
```

```sh
!mkdir -p /content/poc0_data
!mv poc0_corpus.jsonl /content/poc0_data/poc0_corpus.jsonl
!python - <<'PY'
import json
path = "/content/poc0_data/poc0_corpus.jsonl"
count = sum(1 for _ in open(path, encoding="utf-8"))
print({"jsonl_records": count})
PY
```

Local command used before upload:

```sh
./.venv/bin/python scripts/gen_corpus.py \
  --out /tmp/poc0_corpus.jsonl \
  --min-nodes 6 --max-nodes 20 \
  --edge-probs 0.10,0.15,0.20,0.25,0.30 \
  --seeds-per-config 1000
```

Cell 3, convert JSONL to ArrayRecord on Colab Linux:

```sh
!python -m poc0.array_record_data convert \
  --jsonl /content/poc0_data/poc0_corpus.jsonl \
  --out-dir /content/poc0_data/arrayrecord \
  --max-seq-len 80 \
  --split-seed 20260707
```

Cell 4, train:

```sh
!python -m poc0.train \
  --data-dir /content/poc0_data/arrayrecord \
  --workdir /content/poc0_runs/poc0_t4 \
  --batch-size 256 \
  --steps 3000 \
  --warmup-steps 200 \
  --peak-lr 3e-4 \
  --end-lr 3e-5 \
  --weight-decay 0.01 \
  --eval-every 100 \
  --ckpt-every 500 \
  --seed 20260709
```

Cell 5, eval:

```sh
!python -m poc0.eval \
  --checkpoint /content/poc0_runs/poc0_t4/checkpoints/final \
  --jsonl /content/poc0_data/poc0_corpus.jsonl \
  --manifest /content/poc0_data/arrayrecord/manifest.json \
  --out-dir /content/poc0_runs/poc0_t4/eval \
  --n-samples 1000 \
  --temperature 1.0 \
  --seed 20260710
```

Cell 6, inspect summary:

```sh
!cat /content/poc0_runs/poc0_t4/eval/metrics.json
!ls -lh /content/poc0_runs/poc0_t4/eval
```

## 8. TDD Test Plan

Write these tests first in the implementation phase, in this order:

1. `tests/test_poc0_tokenizer.py::test_vocab_ids_are_exact` - the 41-token
   vocab has the exact ids listed in this plan.
2. `tests/test_poc0_tokenizer.py::test_record_to_example_adds_bos_and_masks_to_stop` - a raw token list becomes padded `input_ids`, `target_ids`, and `loss_mask` with loss through the final STOP only.
3. `tests/test_poc0_tokenizer.py::test_tokenizer_rejects_unknown_token_and_pointer_overflow` - unknown tokens and `ptr_32` fail before training.
4. `tests/test_poc0_data.py::test_split_is_seeded_and_stable` - the same records and split seed always produce the same train/val ids with no overlap.
5. `tests/test_poc0_data.py::test_in_memory_source_yields_batch_contract` - in-memory batches have shapes `[batch, 80]`, required dtypes, deterministic shuffle, and finite/repeated iteration behavior.
6. `tests/test_poc0_array_record_data.py::test_arrayrecord_smoke_matches_in_memory_source_or_skips` - on Linux with deps, conversion plus Grain source matches in-memory batches; off-Linux it skips.
7. `tests/test_poc0_model.py::test_model_init_logits_shape_and_param_count` - model init returns logits `[batch, 80, 41]` and parameter count is in the 2-4M band.
8. `tests/test_poc0_model.py::test_causal_mask_blocks_future_positions` - changing a future token cannot change logits at an earlier position.
9. `tests/test_poc0_train.py::test_masked_loss_ignores_padding` - changing padded targets does not change loss when `loss_mask` is false.
10. `tests/test_poc0_train.py::test_train_step_is_deterministic_for_fixed_seed` - one tiny train step produces stable loss/params for fixed init and batch.
11. `tests/test_poc0_sample.py::test_depth_zero_stop_terminates_successfully` - a scripted model that emits top-level STOP returns a successful sample.
12. `tests/test_poc0_sample.py::test_stop_inside_open_level_does_not_terminate` - STOP at positive depth continues until CLOSE then top-level STOP.
13. `tests/test_poc0_sample.py::test_length_cap_without_top_level_stop_is_invalid` - reaching `MAX_SEQ_LEN` without depth-0 STOP returns `length_cap`.
14. `tests/test_poc0_stats.py::test_stats_for_skeleton_matches_corpus_stats_shape` - stats from `decode(tokens)` match expected `_stats_for` fields for hand-built examples.
15. `tests/test_poc0_stats.py::test_total_variation_distance_uses_union_of_bins` - TVD is correct for disjoint and partially overlapping histograms.
16. `tests/test_poc0_eval.py::test_eval_validity_uses_decode_and_sampler_termination` - validity requires both depth-0 STOP termination and `decode()` success.
17. `tests/test_poc0_eval.py::test_eval_novelty_counts_only_valid_samples` - novelty and duplicate-of-train rates exclude invalid samples from the denominator.
18. `tests/test_poc0_eval.py::test_eval_report_contains_required_metrics` - metrics output includes validity, novelty, duplicate rate, per-stat TVD, and histogram counts.

## 9. Staged Delegation Breakdown

Stage 1: Vocab, examples, split, and in-memory data

- Scope: `poc0/constants.py`, `poc0/tokenizer.py`, common objects in
  `poc0/data.py`, and in-memory source.
- Acceptance: tokenizer and data tests pass locally; batches are deterministic
  and match the exact vocab/mask contracts.

Stage 2: Linux ArrayRecord/Grain adapter and Colab README

- Scope: `poc0/array_record_data.py`, manifest format, conversion CLI, Grain
  source, Linux skip behavior, and `poc0/README.md` Colab sequence.
- Acceptance: off-Linux tests skip cleanly; on Linux smoke test proves
  ArrayRecord/Grain batches match the in-memory source for the same records.

Stage 3: Model, train step, schedule, and Orbax checkpoints

- Scope: `poc0/model.py`, `poc0/train.py`, `poc0/checkpoints.py`.
- Acceptance: model/train tests pass, parameter count is 3.19M, masked loss is
  correct, and a tiny local train run writes/restores an Orbax checkpoint.

Stage 4: Sampling and eval metrics

- Scope: `poc0/sample.py`, `poc0/stats.py`, `poc0/eval.py`.
- Acceptance: sampling termination tests pass; eval reports decode-validity,
  novelty, duplicate-of-train rate, histogram overlays, and TVD metrics from
  valid samples.

Stage 5: End-to-end Colab smoke and final verification

- Scope: run the full Colab sequence on the generated corpus with default
  hyperparameters, then clean up docs and command examples if needed.
- Acceptance: training finishes within 30 minutes on T4, final Orbax checkpoint
  exists, eval samples N=1000 at temperature 1.0, and the report includes the
  three PoC-0 acceptance criteria.
