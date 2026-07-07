# PoC-0 Training — Upstream Spec (requirements level)

Decisions fixed here are binding; everything below "left to implementation"
is the implementer's choice, to be recorded in
`docs/plans/poc0_training_impl.md` before coding.

## Goal

Train a small decoder-only Transformer on the PoC-0 token corpus and
verify that it (a) learns the grammar, (b) reproduces the structural
distribution, (c) does not merely memorize. This is a sanity check of the
serialization + modeling direction, not a quality benchmark.

## Fixed decisions

### Stack
- **JAX / Flax (linen) / Optax**, checkpointing with **Orbax**.
- Data loading with **Grain**; on-disk training format **ArrayRecord**.
- Python 3.13, managed with `uv`. `cfg_reducer` core stays stdlib-only.

### Platform split (hard constraint)
- Local dev machine is **macOS (arm64)**. `array_record` ships
  manylinux-only wheels; Grain's macOS support is not to be relied on.
- Therefore: **corpus generation stays local** (existing
  `scripts/gen_corpus.py`, stdlib, JSONL). **JSONL → ArrayRecord
  conversion, the Grain input pipeline, and training run on Colab
  (Linux)**.
- All *logic* (tokenizer, model, train step, sampling loop, eval metrics)
  must be locally testable on macOS with `jax[cpu]` + `flax` + `optax`.
  The Grain/ArrayRecord layer is a thin adapter behind an interface that
  tests replace with an in-memory source; its real path gets a smoke test
  that auto-skips off-Linux.
- Dependencies: new uv dependency group `train` (jax, flax, optax,
  orbax-checkpoint) usable on macOS; `grain` and `array-record` guarded
  with `sys_platform == 'linux'` environment markers so `uv sync` keeps
  working locally.

### Corpus
- Source: `scripts/gen_corpus.py --min-nodes 6 --max-nodes 20
  --edge-probs 0.10,0.15,0.20,0.25,0.30 --seeds-per-config 1000`
  → ~20.7k unique records (measured 2026-07-07; regenerates in ~20 s,
  do not commit data files).
- Train/val split 95/5, seeded random by record. Canonical serialization
  already guarantees exact-string dedup = isomorphism dedup, so the val
  set is structurally disjoint from train by construction.

### Tokenization
- Fixed vocab, no learned tokenizer: `PAD`, `BOS`, `ADD_ENTRY`,
  `ADD_LINEAR`, `ADD_MERGE`, `ADD_LOOP`, `OPEN`, `CLOSE`, `STOP`,
  `ptr_0 … ptr_{K-1}`. `K` = max ptr index in the corpus + safety margin
  (implementer picks margin, records the value).
- **No separate EOS token.** A stream is complete exactly when a `STOP`
  is emitted at nesting depth 0 (OPEN/CLOSE balance). Sampling terminates
  on that condition; hitting the length cap first counts as an invalid
  sample. Loss is computed on tokens after `BOS` up to and including the
  final `STOP`, with padding masked out.

### Model
- Decoder-only Transformer (causal), **~2–4M params** (guidance:
  d_model 256, 4 layers, 4 heads — implementer may tune within the size
  band). Learned positional embeddings are fine at these lengths.
- Max sequence length: 80 (corpus max is 65 tokens + BOS + margin).

### Training
- Next-token cross-entropy, AdamW, warmup + cosine decay.
- Must train to convergence in **≤ ~30 min on a Colab T4**; batch size /
  step count are implementation choices under that budget.
- Deterministic seeding throughout (data split, init, sampling).
- Periodic val loss; final checkpoint saved via Orbax.

### Evaluation (the PoC-0 acceptance criteria)
Sample N=1000 sequences at temperature 1.0 from the trained model, then
report:
1. **Grammatical validity**: fraction accepted by
   `cfg_reducer.linearize.decode()` (plus the depth-0 STOP termination
   rule). Target: ≥ 99% — the grammar is small; near-perfect syntax is
   the pass bar.
2. **Novelty**: fraction of *valid* samples whose exact token string is
   not in the train set. Report both novelty rate and duplicate-of-train
   rate (memorization signal). No fixed target, but ~0% novelty fails
   the PoC.
3. **Distribution match**: for valid samples, compute per-sequence stats
   mirroring `scripts/gen_corpus.py::_stats_for` but derived from the
   decoded `Skeleton` (n_motifs, kind counts, top-level depth/max_width,
   loop_nest, n_tokens); compare histograms against the train set
   (visual overlay + a simple numeric summary per stat, e.g. total
   variation distance). Judgment is qualitative at PoC-0.

The eval must run both on Colab (post-training) and locally against a
checkpoint (CPU) for inspection.

## Deliverables (implementation phase)
1. `docs/plans/poc0_training_impl.md` — detailed plan written BEFORE code.
2. Code under a new top-level package (suggested `poc0/`): tokenizer,
   dataset adapter (in-memory + Grain/ArrayRecord), model, train script,
   sample/eval script, JSONL→ArrayRecord converter (Linux-only path).
3. Tests (t-wada TDD: tests written first) covering tokenizer round-trip,
   masking/loss shapes, sampling termination rule, eval metrics, split
   determinism — all green locally on macOS CPU; Grain/ArrayRecord smoke
   test skipped off-Linux.
4. A short `poc0/README.md` with the exact Colab cell sequence
   (install pins → upload/generate corpus → convert → train → eval).
5. `uv run ty check` clean; existing test suite untouched and green.

## Out of scope for PoC-0
- Graph-conditioned encoding, PE choices (PoC-1 items).
- WL tie-break fix (`docs/plans/wl_tiebreak_design.md`, deferred).
- Hyperparameter search beyond hitting the time budget.
