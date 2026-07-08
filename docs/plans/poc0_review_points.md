# PoC-0 Implementation — Review Points for Final Review

Scope: commits `156236b..da11e25` (train dependency group, GPT-5.5 plan,
Stages 1–4 of the PoC-0 training pipeline, and the PYTHONHASHSEED
determinism fix in cfg_reducer). Binding documents:
`docs/plans/poc0_training_spec.md` (upstream requirements) and
`docs/plans/poc0_training_impl.md` (detailed plan). Review is expected to
hunt independently for correctness bugs beyond this list; the items below
are what the in-session review already flagged and wants a second opinion
on, plus known accepted trade-offs that should NOT be re-litigated unless
they hide a real defect.

## Items wanting a second opinion (accumulated during staged review)

1. **Grain usage is nominal** (`poc0/array_record_data.py`):
   `GrainArrayRecordDataset` materializes all records in memory and
   batches in pure Python; Grain appears only as a fallback ArrayRecord
   reader. This constructively guarantees the "byte-identical to
   InMemoryTokenDataset" acceptance, and is fine at 20.7k records, but it
   deviates from the plan wording ("builds a Grain source/data-loader").
   Question: acceptable for PoC-0 (we think yes); anything that breaks
   when the corpus grows?
2. **ArrayRecord API probing**: `_arrayrecord_writer/_reader` try
   multiple constructor signatures via getattr/TypeError fallbacks
   because the API could not be exercised on macOS. Fragile? The Linux
   smoke test (Stage 5, Colab) exercises it — review whether the probing
   can silently pick a wrong-but-working constructor.
3. **Empty stream is grammatical**: `decode(["STOP"])` accepts (empty
   top level), so a bare `STOP` sample counts as VALID with 0 motifs and
   n_tokens=1 — observed in the local smoke eval (bin "0"/"1" in the
   histograms). The corpus never contains it (min 16 tokens). Should eval
   report it as its own category (e.g. `empty_stream` count) instead of
   letting it inflate validity/novelty? Recommend: report, don't reject.
4. **`record_to_example([])`** silently yields an all-PAD example with an
   all-False mask instead of erroring (tokenizer edge; unreachable from
   real corpora — but a loud error would be cheaper than a silent
   degenerate example if a data bug ever produces empty token lists).
5. **`iter_batches` duplication**: copy-pasted between
   `InMemoryTokenDataset` and `GrainArrayRecordDataset` — extract-shared
   candidate (quality, not correctness).
6. **Reviewer-applied type fixes in `poc0/eval.py`**: `_as_int`/
   `_as_float` helpers and one `cast` were added by the reviewing session
   (not Codex) to clear ty diagnostics after the Stage 4 job died on an
   upstream capacity error. Sanity-check these did not alter semantics
   (they should only narrow JSON-derived `object` values).
7. **Sampler efficiency**: `sample_tokens` re-runs a full forward over
   the padded length-80 prefix per generated token, batch size 1 —
   O(len²) per sample, no KV cache, no batched sampling. At N=1000 ×
   ≤79 tokens on a T4 this is tolerable but slow; flag if it threatens
   the 30-minute budget (eval is outside the training budget, so this is
   a nuisance, not a violation).
8. **LR logging nuance** (`train.py`): the printed `lr` is
   `schedule(state.step)` AFTER `apply_gradients`, i.e. the rate of the
   NEXT step, not the one just applied. Cosmetic; confirm no test/logic
   depends on it.

## Accepted decisions — do not re-litigate (context, with rationale)

- **No EOS token; depth-0 STOP terminates; length-cap = invalid** — spec
  §Tokenization, deliberate: the model must learn the grammar.
- **Sampler masks only PAD/BOS** — deliberate; grammar constraints are
  not enforced at decode time for PoC-0.
- **Dropout 0.0, tied LM head, learned positional embeddings, 3,190,528
  params** — plan §2, verified exactly.
- **In-process canonicality self-check + 3.2% discard** — WL tie-break
  fix is designed and deferred (docs/plans/wl_tiebreak_design.md);
  corpus correctness is guaranteed by the filter.
- **Cross-process determinism fix** (tarjan child sort, smallest-terminal
  SCC pick) — md5-verified byte-identical sweeps; new subprocess test
  pins PYTHONHASHSEED independence.
- **Platform split** (corpus local/macOS, ArrayRecord+train on
  Colab/Linux) — array-record ships manylinux wheels only.

## Verification state at review time

- `./.venv/bin/python -m pytest tests/ -q`: 65 passed, 1 skipped (the
  skip is the Linux-only ArrayRecord/Grain smoke test).
- `./.venv/bin/ty check`: clean.
- Local end-to-end: gen_corpus (tiny) → train 20 steps (loss 3.93→1.44,
  deterministic across machines/invocations to full float precision) →
  eval 50 samples → metrics.json + 9 PNGs. Validity 4% at 20 steps
  (expected garbage; mechanics verified, quality comes from Stage 5).
