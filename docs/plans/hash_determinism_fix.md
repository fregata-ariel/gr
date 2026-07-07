# Cross-Process Determinism Fix (PYTHONHASHSEED independence)

## Problem

The reduction → encode pipeline is deterministic only *within* a process.
Measured 2026-07-08: two plain runs of the standard `gen_corpus` sweep
(n=6–20, p=0.10–0.30, 1000 seeds/config) produced different outputs
(written=20687 vs 20690, different md5), while two runs with
`PYTHONHASHSEED=0` were byte-identical. Root cause: Python's per-process
string-hash salt changes `set` iteration order somewhere in the pipeline,
so the same graph can encode to different token streams in different
processes. This breaks the core promise that canonical serialization makes
isomorphism = string equality *across* runs/machines, and makes corpora
non-reproducible.

The in-process canonicality self-check cannot catch this: both encodes in
the check share the process's hash salt.

## Suspected leak sites (verify, do not assume)

- `cfg_reducer/algorithm.py::tarjan_scc` — iterates `succ_fn(node)`
  (a set from `engine.successors`) when collecting children; child visit
  order changes DFS index assignment and the *order* of the returned SCC
  list.
- `_pick_terminal_scc` — returns the first terminal SCC in that list;
  with multiple terminal SCCs the pick becomes hash-order-dependent,
  changing reduction step order and hence `mg.motifs` order (the
  `motif_positions` tie-break in `linearize.canonical_order`).
- Audit ALL other set/dict iterations on the reduction path
  (`algorithm.py`, `motif.py`, `metagraph.py`, `linearize.py`,
  `engine.py`) whose order can reach the Op history, motif order, or
  token stream. Iterations that are immediately `sorted(...)` or
  `min(...)`/`max(...)`-reduced are safe.

## Fix policy

Make iteration order explicit (e.g. `sorted(...)`) at each leak site.
Do NOT fix by requiring `PYTHONHASHSEED` — the library must be
deterministic regardless of environment. Keep changes minimal; no
behavior redesign. Note: fixing iteration order MAY change which token
stream is chosen for some graphs relative to past corpora — that is
acceptable (corpora are regenerable in seconds); same-process seeded
determinism tests must still pass.

## Tests (TDD — write first)

1. **Cross-hash-seed invariance (the real test):** for several
   `build_cfg` graphs (mixed n, seeds), run the full
   reduce → extract → build → encode pipeline in subprocesses with
   `PYTHONHASHSEED` set to different values (e.g. 0, 1, 2) via
   `subprocess.run([sys.executable, "-c", ...], env=...)`, and assert
   identical token streams across hash seeds. Must FAIL before the fix
   (use graph parameters confirmed to expose the leak — hunt with a
   quick sweep if needed).
2. Existing suite (47 tests) unchanged and green.

## Acceptance

- New cross-hash-seed test green; full suite green; `ty check` clean.
- Manual check: two plain (no PYTHONHASHSEED) runs of the standard
  1000-seeds sweep produce byte-identical JSONL (same md5).
