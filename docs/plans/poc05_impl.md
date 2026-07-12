# PoC-0.5 Implementation Plan

This plan is bound by `docs/plans/poc05_spec.md`. It is planning only: the
implementation phase must add grammar-constrained sampling as an opt-in path,
scale the generated corpus, and keep PoC-0's default sampling behavior
unchanged when `--constrained` is absent.

The legality rules below are derived from `cfg_reducer/linearize.py::_Parser`.
The tracker must mirror `decode()` acceptance over the fixed PoC-0 vocabulary;
it must not add corpus-convention constraints such as kind arity.

## Spec Conflicts

No implementation-blocking conflict was found.

Source nuance to preserve in implementation and tests: `_Parser.parse_parents()`
does not enforce an intrinsic maximum pointer index. It accepts any token string
that starts with `ptr_`, parses as an integer, and satisfies the current-level
`parent < n_items` and strict ascending checks. The PoC-0 sampler, tokenizer,
and logits domain are fixed to `VOCAB`, which only contains `ptr_0` through
`ptr_31`. Therefore the tracker should enforce `k <= 31` as a vocabulary-domain
candidate bound, not as a claim that `decode()` would reject every arbitrary
`ptr_32` string in a level with more than 32 completed items.

## 1. Legality Tracker

Add a new module `poc0/grammar.py`. It owns a small incremental parser state
for sampling masks. It should not import or call `_Parser` in production code;
tests will compare it against `decode()` as the oracle.

Planned API:

```text
GrammarTracker.initial() -> GrammarTracker
GrammarTracker.legal_token_ids() -> tuple[int, ...]
GrammarTracker.legal_mask() -> np.ndarray  # bool shape (VOCAB_SIZE,)
GrammarTracker.is_legal_token(token: str) -> bool
GrammarTracker.is_legal_token_id(token_id: int) -> bool
GrammarTracker.advance_token(token: str) -> GrammarTracker
GrammarTracker.advance_token_id(token_id: int) -> GrammarTracker
GrammarTracker.is_terminal -> bool
```

`advance_*` returns a new tracker and raises `ValueError` when the token is not
legal in the current state. `legal_mask()` is a host NumPy bool mask so
`poc0/grammar.py` has no JAX dependency; `poc0/sample.py` can convert it with
`jnp.asarray(...)` only on the constrained path.

State data:

| Field | Meaning |
|---|---|
| `stack` | Current parse-level frames. Each frame stores `items_completed` for that level. The first frame is the top level. |
| `phase` | One of `ITEM`, `PARENTS_NONLOOP`, `PARENTS_LOOP`, `EXPECT_CLOSE`, `DONE`. |
| `pending_n_items` | In a parents phase, the number of completed items in the current level before the pending item. This is the pointer upper bound. |
| `previous_ptr` | In a parents phase, the last accepted pointer index, initialized to `-1`. |

When an `ADD_LOOP` parent list accepts `OPEN`, the loop item is appended to the
parent level before the child level starts, matching `_Parser.parse_level()`.
The tracker increments the parent frame's `items_completed`, pushes a new child
frame with `items_completed = 0`, and enters `ITEM` for the child. When a child
level accepts `STOP`, the tracker pops the child frame and enters
`EXPECT_CLOSE`; only after a legal `CLOSE` does it return to `ITEM` in the
parent level.

Token classes used by the table:

| Class | Tokens |
|---|---|
| `ADD_NONLOOP` | `ADD_ENTRY`, `ADD_LINEAR`, `ADD_MERGE` |
| `ADD_LOOP` | `ADD_LOOP` |
| `PTR(k)` | In-vocabulary `ptr_k`, where `0 <= k <= 31` |
| `OPEN` | `OPEN` |
| `STOP` | `STOP` |
| `CLOSE` | `CLOSE` |
| `SPECIAL` | `PAD`, `BOS`, and any out-of-vocabulary token id |

State machine table:

| State | `ADD_NONLOOP` | `ADD_LOOP` | `PTR(k)` | `OPEN` | `STOP` | `CLOSE` | `SPECIAL` |
|---|---|---|---|---|---|---|---|
| `ITEM_TOP(n)` | Legal -> `PARENTS_NONLOOP(top, n, -1)` | Legal -> `PARENTS_LOOP(top, n, -1)` | Illegal; item boundary expects ADD or STOP | Illegal; `_Parser` only allows OPEN after loop parents | Legal -> `DONE` | Illegal; unexpected top-level CLOSE | Illegal |
| `ITEM_CHILD(n)` | Legal -> `PARENTS_NONLOOP(child, n, -1)` | Legal -> `PARENTS_LOOP(child, n, -1)` | Illegal; item boundary expects ADD or STOP | Illegal; `_Parser` only allows OPEN after loop parents | Legal -> pop child and enter `EXPECT_CLOSE(parent)` | Illegal; `_Parser` reports CLOSE without preceding child-level STOP | Illegal |
| `PARENTS_NONLOOP(L, n, prev)` | Legal: complete pending item, increment current frame to `n + 1`, then consume this ADD -> `PARENTS_NONLOOP(L, n + 1, -1)` | Legal: complete pending item, increment current frame to `n + 1`, then consume this ADD -> `PARENTS_LOOP(L, n + 1, -1)` | Legal iff `prev < k < n`; stay in `PARENTS_NONLOOP(L, n, k)`. Otherwise illegal for self/forward, duplicate, non-ascending, or out-of-vocab pointer. | Illegal; `_Parser` explicitly rejects OPEN after non-loop parents | Legal: complete pending item, increment current frame to `n + 1`, then STOP ends the level -> `DONE` if top else `EXPECT_CLOSE(parent)` | Illegal; after completing the pending item, CLOSE is still illegal at an item boundary | Illegal |
| `PARENTS_LOOP(L, n, prev)` | Illegal; ADD_LOOP parents must end with OPEN | Illegal; ADD_LOOP parents must end with OPEN | Legal iff `prev < k < n`; stay in `PARENTS_LOOP(L, n, k)`. Otherwise illegal for self/forward, duplicate, non-ascending, or out-of-vocab pointer. | Legal: complete loop item, increment current frame to `n + 1`, push child frame -> `ITEM_CHILD(0)` | Illegal; ADD_LOOP parents must end with OPEN | Illegal; ADD_LOOP parents must end with OPEN | Illegal |
| `EXPECT_CLOSE(parent)` | Illegal | Illegal | Illegal | Illegal | Illegal | Legal -> `ITEM_TOP(parent_n)` or `ITEM_CHILD(parent_n)`, depending on parent stack depth | Illegal |
| `DONE` | Illegal; sampler should already have stopped | Illegal | Illegal | Illegal | Illegal | Illegal | Illegal |

Important consequences:

- Pointer bounds are per current level, not global. Child levels start with
  zero completed items, and parent counts resume after the matching `CLOSE`.
- `PTR(k)` is legal only in a parents phase, immediately after an `ADD_*` token
  or another legal pointer in the same parents list.
- `k < n` rejects self and forward references. `k > prev` gives ascending and
  unique parents; `k == prev` is duplicate, `k < prev` is non-ascending.
- `STOP` is legal at parse item boundaries. In `PARENTS_NONLOOP`, an incoming
  `STOP` first terminates the non-loop parents list, completes the item, and is
  then consumed by the outer parse loop as the boundary `STOP`. This is exactly
  how `_Parser.parse_parents()` leaves a non-pointer token unconsumed.
- The same delimiter behavior makes `ADD_*` legal after a non-loop parents
  list: it completes the previous item and starts the next item.
- `ADD_LOOP` is different: after its parents list, the next token must be
  `OPEN`. `ADD_*`, `STOP`, and `CLOSE` are illegal until `OPEN` is emitted.
- After a child level's `STOP`, the only legal next token is `CLOSE`.
- `PAD` and `BOS` are never legal emitted grammar tokens and stay masked in
  every non-terminal state.
- Do not encode kind arity. For example, if pointers satisfy the table,
  `ADD_ENTRY ptr_0` after a prior completed item is decode-legal even if the
  corpus generator does not normally emit it.

## 2. `sample_tokens` Mask Integration

Extend the sampler signature to:

```text
sample_tokens(*, logits_fn, temperature, rng, constrained: bool = False)
```

Default behavior must remain byte-identical to PoC-0. In the
`constrained=False` branch, preserve the current operation order:

1. Build `prefix_token_ids = [BOS_ID]`.
2. Call `logits_fn(jnp.asarray(prefix_token_ids, dtype=jnp.int32))`.
3. Call the existing `_masked_logits()` to set only `PAD` and `BOS` to
   `-inf`.
4. Split the JAX PRNG once per generated token.
5. Sample with `jax.random.categorical(sample_rng, logits / temperature)`.
6. Update the emitted tokens and the existing depth-based termination checks.

Do not construct a tracker, compute a grammar mask, add RNG splits, change
temperature handling, or alter invalid-reason behavior on the unconstrained
path.

For `constrained=True`:

1. Create `tracker = GrammarTracker.initial()` before the sampling loop.
2. For each step, compute logits and apply `_masked_logits()` as today.
3. Convert `tracker.legal_mask()` to a JAX bool array and set every illegal
   token id to `-jnp.inf` before dividing by temperature.
4. Sample from the masked logits with the same one-split-per-token pattern.
5. Convert the sampled id to a token and advance the tracker with
   `tracker = tracker.advance_token_id(next_token_id)`. A `ValueError` here is
   an implementation bug; the mask should make it unreachable.
6. Keep the existing `depth` update for `OPEN` and `CLOSE` so the returned
   `SampleResult.depth` stays compatible with PoC-0.
7. Return success on `STOP` only when the existing depth counter is zero.
   `STOP` inside a child level is legal but transitions the tracker to
   `EXPECT_CLOSE` and sampling continues.
8. If the loop reaches `MAX_SEQ_LEN - 1`, return `length_cap` as today.

With a correct tracker, constrained samples can fail only by length cap. The
existing `negative_depth` branch remains present for unconstrained mode and as
a defensive invariant, but constrained tests should prove it is unreachable.
`decode_error` is still assigned by eval if a supposedly successful sample does
not decode; the constrained property tests should prove that is unreachable
except for implementation bugs.

The mask is applied before temperature:

```text
masked = _masked_logits(raw_logits)
if constrained:
    masked = jnp.where(grammar_mask, masked, -jnp.inf)
next_id = categorical(masked / temperature)
```

## 3. Eval CLI Wiring

Extend `poc0/eval.py` as follows:

| Surface | Change |
|---|---|
| CLI | Add `--constrained` as `store_true`, default `False`. |
| `evaluate_checkpoint()` | Add `constrained: bool = False` and pass it to `sample_tokens(...)`. |
| Metrics | Add `report["constrained"] = bool(constrained)` before `write_eval_report()`, so `metrics.json` and stdout both contain the field. |
| Dump samples | Keep the existing JSONL dump schema unchanged. `--dump-samples` should work in both constrained and unconstrained modes and must not change metrics except for sampling consequences. |

Default CLI calls without `--constrained` remain unconstrained. The metrics file
will gain the required `"constrained": false` field, but sampled token streams,
sample classifications, RNG usage, and dump content for default sampling should
match PoC-0 behavior.

The evaluator continues to call `decode()` for successful sampler returns. This
keeps unconstrained validity honest and provides a guardrail for constrained
mode. Empty streams (`["STOP"]`) remain decode-accepted but are still counted by
the existing `empty_stream_count` path rather than as valid non-empty samples.
The mask must not forbid top-level `STOP` solely to avoid empty streams, because
`decode()` accepts it.

## 4. TDD Test Plan

Write tests before implementation in this order.

1. `tests/test_poc0_grammar.py::test_initial_item_boundary_legality`:
   initial top-level state allows `ADD_ENTRY`, `ADD_LINEAR`, `ADD_MERGE`,
   `ADD_LOOP`, and `STOP`; rejects pointers, `OPEN`, `CLOSE`, `PAD`, and `BOS`.
2. `tests/test_poc0_grammar.py::test_nonloop_parent_delimiters_match_parser`:
   after a non-loop ADD, `ADD_*` and `STOP` are legal delimiters that complete
   the current item; `OPEN` and `CLOSE` are illegal.
3. `tests/test_poc0_grammar.py::test_loop_parent_list_requires_open`:
   after `ADD_LOOP` and after legal loop-parent pointers, only more legal
   pointers or `OPEN` are allowed; `ADD_*`, `STOP`, and `CLOSE` are rejected.
4. `tests/test_poc0_grammar.py::test_pointer_bounds_are_current_level_ascending_unique`:
   verify `k < items_completed`, strict ascending order, duplicate rejection,
   self/forward rejection, and in-vocabulary `ptr_0..ptr_31` candidate bounds.
5. `tests/test_poc0_grammar.py::test_child_stop_requires_close_only`:
   after `ADD_LOOP OPEN STOP`, the only legal token is `CLOSE`; after `CLOSE`,
   parent-level item-boundary legality resumes.
6. `tests/test_poc0_grammar.py::test_child_level_pointer_counts_reset_and_parent_counts_restore`:
   child levels start with zero completed items, reject parent pointers inside
   the child until child items exist, and restore the parent count after close.
7. `tests/test_poc0_grammar.py::test_terminal_state_has_no_legal_tokens`:
   top-level `STOP` enters `DONE`, and every token id is illegal afterward.
8. `tests/test_poc0_grammar.py::test_no_kind_arity_constraints`:
   construct decode-accepted but corpus-unusual examples such as `ADD_ENTRY`
   with a legal parent pointer and verify the tracker allows them.
9. `tests/test_poc0_grammar.py::test_legal_mask_shape_and_ids`:
   `legal_mask()` has shape `(VOCAB_SIZE,)`, dtype bool, and matches
   `legal_token_ids()` exactly.
10. Property obligation A1,
    `tests/test_poc0_grammar_properties.py::test_mask_implies_decode_accepts`:
    run randomized/adversarial constrained sampling where illegal logits are
    often boosted above legal logits. For every non-length-cap sample that
    terminates successfully, assert `decode(result.tokens)` accepts. For every
    failed constrained sample, assert `invalid_reason == "length_cap"`.
11. Property obligation A2,
    `tests/test_poc0_grammar_properties.py::test_mask_never_overblocks_decode_oracle`:
    on randomized legal prefixes, compare the tracker legal set against a
    decode-derived prefix oracle over every `VOCAB` token id. The oracle runs
    `decode(prefix + [candidate])`; the candidate is legal when decode succeeds
    or when the parser error position is after the candidate position, meaning
    `_Parser` consumed the candidate and only failed because the stream needs
    future tokens. An error at the candidate position means the candidate is
    illegal. Stop rollouts at top-level terminal `STOP`.
12. `tests/test_poc0_sample.py::test_unconstrained_sampling_regression_matches_poc0_snapshot`:
    with fixed deterministic logits and fixed seeds, assert the exact
    unconstrained `SampleResult` values observed before the change. This guards
    byte-for-byte behavior of default sampling.
13. `tests/test_poc0_sample.py::test_constrained_mask_blocks_negative_depth_and_decode_errors`:
    use logits that prefer illegal `CLOSE`, `OPEN`, or pointer tokens at
    specific states and verify constrained sampling chooses legal alternatives.
14. `tests/test_poc0_eval.py::test_eval_parse_args_constrained_default_false`:
    `--constrained` defaults to false and parses to true when present.
15. `tests/test_poc0_eval.py::test_evaluate_checkpoint_passes_constrained_to_sampler`:
    monkeypatch `sample_tokens` and assert `constrained` is forwarded as false
    by default and true with the flag.
16. `tests/test_poc0_eval.py::test_metrics_json_records_constrained_field`:
    metrics contain `"constrained": false` or true as appropriate.
17. `tests/test_poc0_eval.py::test_dump_samples_interaction_with_constrained`:
    `--dump-samples` writes the same schema in constrained mode, and enabling
    dumps does not change metrics for the same scripted sample stream and same
    `constrained` value.

## 5. Corpus Probe

Probe command run during planning:

```sh
./.venv/bin/python scripts/gen_corpus.py \
  --out /tmp/poc05_probe.jsonl \
  --min-nodes 6 --max-nodes 20 \
  --edge-probs 0.10,0.15,0.20,0.25,0.30 \
  --seeds-per-config 5000
```

Fresh probe stderr:

```text
generated=375000 discarded=11487 deduped=267525 written=95988
```

Measured from `/tmp/poc05_probe.jsonl`:

| Value | Measurement |
|---|---:|
| Record count | 95,988 |
| Min raw token length, no BOS | 16 |
| Max raw token length, no BOS | 67 |
| Max length with BOS | 68 |
| Record id at max raw length | `n20_p0.3_s4888` |
| Max pointer index observed | 17 |
| Observed pointer token range | `ptr_0` through `ptr_17` |

Bounds verdict:

| Bound | Status |
|---|---|
| Pointer vocabulary `ptr_0..ptr_31` | Fits: max observed pointer index is 17. |
| `MAX_SEQ_LEN = 80` total positions including BOS | Fits: max BOS-prepended length is 68, leaving 12 positions of margin. |

No corpus-bound spec conflict is present. Do not grow the vocabulary or
sequence length for PoC-0.5.

## 6. Implementation Split

Job 1: legality tracker and sampler integration.

Scope:

- Add `poc0/grammar.py`.
- Add grammar unit tests and both property/fuzz tests.
- Add the optional `constrained` parameter to `sample_tokens`.
- Keep the unconstrained code path behavior-identical.

Acceptance criteria:

- `tests/test_poc0_grammar.py` and grammar property tests pass.
- Existing PoC-0 sample tests pass.
- The unconstrained snapshot regression passes with exact tokens, success
  flags, invalid reasons, and depth values.
- Constrained randomized/adversarial sampling produces only decode-accepted
  successful streams or `length_cap` failures.

Job 2: eval wiring and protocol updates.

Scope:

- Add `--constrained` to `python -m poc0.eval`.
- Thread the flag through `evaluate_checkpoint()` to `sample_tokens`.
- Add `"constrained"` to metrics.
- Keep dump-samples schema stable.
- Add eval wiring tests.
- Update user-facing Colab/run notes if implementation work includes README
  changes.

Acceptance criteria:

- Eval tests pass for default and constrained modes.
- CLI without `--constrained` forwards `constrained=False`.
- CLI with `--constrained` forwards `constrained=True`.
- `metrics.json` records the correct boolean in both modes.
- `--dump-samples` works in both modes and does not perturb metrics for a
  scripted sample stream.

## 7. Colab Protocol Delta

Everything else remains per the PoC-0 Colab protocol: same dependency stack,
same model, same hyperparameters, same split rule, same seeds, same 3000-step
training budget, and the same final/best checkpoint locations.

Local corpus generation before upload changes only `--seeds-per-config` and
the output filename:

```sh
./.venv/bin/python scripts/gen_corpus.py \
  --out /tmp/poc05_corpus.jsonl \
  --min-nodes 6 --max-nodes 20 \
  --edge-probs 0.10,0.15,0.20,0.25,0.30 \
  --seeds-per-config 5000
```

Run four eval commands after training, each with `--dump-samples`, for final
and best checkpoints in both modes.

Final checkpoint, unconstrained:

```sh
!python -m poc0.eval \
  --checkpoint /content/poc0_runs/poc05_t4/checkpoints/final \
  --jsonl /content/poc0_data/poc05_corpus.jsonl \
  --manifest /content/poc0_data/arrayrecord/manifest.json \
  --out-dir /content/poc0_runs/poc05_t4/eval_final_unconstrained \
  --n-samples 1000 \
  --temperature 1.0 \
  --seed 20260710 \
  --dump-samples /content/poc0_runs/poc05_t4/eval_final_unconstrained/samples.jsonl
```

Final checkpoint, constrained:

```sh
!python -m poc0.eval \
  --checkpoint /content/poc0_runs/poc05_t4/checkpoints/final \
  --jsonl /content/poc0_data/poc05_corpus.jsonl \
  --manifest /content/poc0_data/arrayrecord/manifest.json \
  --out-dir /content/poc0_runs/poc05_t4/eval_final_constrained \
  --n-samples 1000 \
  --temperature 1.0 \
  --seed 20260710 \
  --constrained \
  --dump-samples /content/poc0_runs/poc05_t4/eval_final_constrained/samples.jsonl
```

Best checkpoint, unconstrained:

```sh
!python -m poc0.eval \
  --checkpoint /content/poc0_runs/poc05_t4/checkpoints/best \
  --jsonl /content/poc0_data/poc05_corpus.jsonl \
  --manifest /content/poc0_data/arrayrecord/manifest.json \
  --out-dir /content/poc0_runs/poc05_t4/eval_best_unconstrained \
  --n-samples 1000 \
  --temperature 1.0 \
  --seed 20260710 \
  --dump-samples /content/poc0_runs/poc05_t4/eval_best_unconstrained/samples.jsonl
```

Best checkpoint, constrained:

```sh
!python -m poc0.eval \
  --checkpoint /content/poc0_runs/poc05_t4/checkpoints/best \
  --jsonl /content/poc0_data/poc05_corpus.jsonl \
  --manifest /content/poc0_data/arrayrecord/manifest.json \
  --out-dir /content/poc0_runs/poc05_t4/eval_best_constrained \
  --n-samples 1000 \
  --temperature 1.0 \
  --seed 20260710 \
  --constrained \
  --dump-samples /content/poc0_runs/poc05_t4/eval_best_constrained/samples.jsonl
```

Report final and best metrics for both modes. The constrained acceptance target
is validity >= 99.9%, with failures expected only from length cap. The
unconstrained runs remain diagnostic and should be used to compare pointer
violation share and validity against the PoC-0 baseline.
