# PoC-0.5 — Grammar-Constrained Sampling + Corpus Scale-Up (upstream spec)

Follows PoC-0 (docs/plans/poc0_training_spec.md, poc0_training_impl.md,
results in runs/poc0_t4_rerun/). PoC-0 verdict: distribution match (TVD
0.006–0.112) and novelty (79.6–91.9%) pass; grammatical validity misses
the ≥99% bar (84.5% final / 80.5% best), with **~85% of decode errors
being ptr-constraint violations** (forward/self reference dominant, then
non-ascending / duplicate; occasional ptr in item position). Decision
(user, 2026-07-10): add grammar-constrained sampling AND scale the corpus.

## Fixed decisions

### A. Decode-aligned constrained sampling (opt-in)
- `poc0/sample.py` gains an incremental legality tracker mirroring
  `cfg_reducer.linearize._Parser`'s acceptance rules exactly, and
  `sample_tokens(..., constrained: bool = False)`.
- **Legality = decode()'s rules, nothing more** (do NOT encode corpus
  conventions like kind arity — decode does not check arity):
  - `ptr_k` legal iff currently inside a parents list (immediately after
    an ADD_* token or another ptr), AND `k` < number of items already
    completed in the CURRENT level, AND `k` > the previous ptr index of
    the current parents list (ascending, unique), AND `k` ≤ 31 (vocab).
  - `OPEN` legal only while in the parents context of an `ADD_LOOP`
    (i.e. immediately after `ADD_LOOP` or its ptrs). After any other
    kind's parents, `OPEN` is illegal.
  - An `ADD_LOOP` parents context MUST end with `OPEN` — `ADD_*`/`STOP`
    are illegal until the `OPEN` is emitted.
  - After a child level's `STOP` (depth returning to the loop item),
    the only legal token is `CLOSE`.
  - `STOP` legal at item boundaries (level start or after a completed
    item), any depth. Depth-0 STOP terminates (unchanged).
  - `PAD`/`BOS` always masked (unchanged).
- Masking sets illegal logits to -inf before temperature/sampling.
  With constrained=True a sample can still be invalid ONLY via length
  cap (report as today).
- `python -m poc0.eval` gains `--constrained`; metrics.json records
  `"constrained": true|false`. Default (flag absent) behavior byte-
  identical to today — unconstrained validity stays the honest metric
  of grammar learning.
- **Correctness obligation (test, not prose)**: property/fuzz tests must
  establish both directions — (1) any token stream produced under the
  mask (driven by adversarial/random logits) is accepted by `decode()`
  (up to length cap); (2) the mask never forbids a token that `decode()`
  would accept at that position (cross-check tracker legality against a
  reference incremental oracle derived from running `decode` on
  prefix+candidate, on randomized rollouts).

### B. Corpus scale-up
- Same sweep ranges, `--seeds-per-config 5000` → measured 2026-07-10:
  **95,988 unique records** (~98 s locally, deterministic). No code
  change; corpus is regenerated, not committed.
- Split rule, tokenizer, vocab (ptr_0..31), MAX_SEQ_LEN 80 unchanged —
  BUT the plan must re-verify max ptr index and max sequence length on
  the 96k corpus (probe it); if either exceeds current bounds, STOP and
  surface (do not silently grow the vocab).

### C. Training/eval protocol (Colab T4, unchanged budget)
- Same hyperparameters, 3000 steps, batch 256 (≈8 epochs at ~91k train
  records vs ~39 before). Same seeds.
- Eval: N=1000, temp 1.0, for BOTH checkpoints (final, best) × BOTH
  modes (unconstrained, constrained) = 4 runs, each with --dump-samples.
- Acceptance: constrained validity ≥ 99.9% (length-cap only);
  unconstrained validity reported (diagnostic — does more data shrink
  the ptr-violation share?); novelty + TVD re-reported.

## Process
1. Detailed plan → `docs/plans/poc05_impl.md` (planning model), including
   the tracker state machine table, test list (TDD order), and a 1–2 job
   implementation split.
2. Implementation jobs (TDD, tests first) per that split.
3. Claude-side review each stage; Colab run driven from the Claude
   session (Codex sandbox cannot run the colab CLI).

## Out of scope
- Pointer-aware architectures (PoC-1 design item).
- WL tie-break (still deferred).
