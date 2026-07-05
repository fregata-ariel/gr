# Reducible Loop Generation Implementation Plan

## Goal

Constrain `build_cfg()`'s spaghetti phase so every generated CFG is
**reducible**: each back edge `u → v` is only added when `v` dominates `u`.
This eliminates multi-entry (irreducible) loops, which currently cause the
canonicality self-check in `scripts/gen_corpus.py` to discard ~46% of
generated graphs overall and ~74% at 18–20 nodes (measured 2026-07-04,
sweep n=6–20, p=0.10–0.30, 25 seeds/config).

Rationale: real CFGs from structured languages are almost always reducible,
so this moves `build_cfg` *closer* to the target (calisp) distribution while
making the reduction trace label-invariant (single-entry SCC ⇒ unique header
⇒ `_identify_header`'s `min()` is trivially canonical, including nested
loops: in a reducible CFG the only entry to a natural loop is its header).

Development follows TDD: tests first, then implementation.

## Files to modify

| File | Action | What |
|------|--------|------|
| `cfg_reducer/gen.py` | Update | reorder spaghetti phase, dominator-gated back edges |
| `tests/test_gen.py` | Extend | reducibility + canonicality property tests |

No other files change. `main.py`, `linearize.py`, `gen_corpus.py` are
untouched.

## 1. `build_cfg()` changes (`cfg_reducer/gen.py`)

Phase 1 (layered DAG construction) is unchanged.

Phase 2 (spaghetti) changes:

**a. Reorder: forward jumps `[B]` BEFORE back edges `[A]`.**
Forward jumps keep the graph acyclic, but a jump added *after* a back edge
could land inside an existing loop body and create a second entry.
Dominators must therefore be computed after all forward edges exist.
Conversely, adding a back edge whose target dominates its source does not
change the dominator relation (every path through `u → v` already passed
`v` to reach `u`), so all back edges can be validated against dominators
computed once on the final acyclic graph.

**b. Compute dominators once, after jumps.**
Add a private helper in `gen.py`:

```python
def _dominators(engine: GraphEngine, entry: str) -> dict[str, set[str]]:
```

Standard iterative dataflow on the acyclic graph:
`dom(entry) = {entry}`; for other nodes `dom(n) = {n} ∪ ⋂ dom(preds(n))`,
iterate to fixpoint. Graphs are ≤ ~20 nodes; the naive set algorithm is
fine. Iterate nodes in a deterministic order (e.g. sorted). Every node is
reachable from `ids[0]` by construction (each node is given at least one
parent; the entry is `ids[0]`).

**c. Back edges `[A]`: dominator-gated target choice.**
Keep `num_loops = max(1, int(num_nodes * 0.15))` attempts. Per attempt:

1. `j = rng.randint(2, num_nodes - 2)` (unchanged draw)
2. candidates = proper dominators of `ids[j]` restricted to
   `ids[0] .. ids[j-2]` (preserves the old "return at least 2 back" rule,
   which also avoids 2-cycles with a direct predecessor), excluding targets
   `v` where the edge `ids[j] → v` already exists
3. if candidates is empty, skip this attempt (no retry — keeps the RNG
   call budget per attempt bounded and deterministic)
4. else `v = rng.choice(sorted(candidates))` and add edge `ids[j] → v`
   (sorted for determinism — candidate sets must not depend on set
   iteration order)

Forward jumps `[B]` keep their existing logic (they cannot create cycles);
only their position moves.

**d. Update the docstring** to state the reducibility guarantee.

Note: same-seed output CHANGES relative to the previous version (RNG call
order moved). That is intended; no test snapshots exact edges. Seeded
determinism (same seed ⇒ same graph) must still hold.

## 2. Tests (written FIRST — TDD, extend `tests/test_gen.py`)

Existing tests in `tests/test_gen.py` and the rest of the suite must keep
passing (the 4 currently-skipped round-trip seeds in `test_linearize.py`
are expected to stop skipping once this lands — do not modify that file).

New tests:

**Reducibility property (the definition, checked independently):**
For `num_nodes in {12, 20}` × `edge_prob in {0.2, 0.5}` × `seed in range(20)`:
build the graph, and with a test-local dominator computation (do NOT import
the helper from `gen.py` — an independent implementation cross-checks it),
assert the graph is reducible: **remove every edge `u → v` whose target
dominates its source (`v ∈ dom(u)`), then assert the remaining graph is
acyclic** (Kahn or DFS).

> Corrected 2026-07-04: an earlier revision of this section demanded
> `v ∈ dom(u)` for every cycle-closing edge (every `u → v` where `v`
> reaches `u`). That predicate is strictly stronger than reducibility —
> in any cycle of length ≥ 2 the two endpoints cannot dominate each
> other, so it effectively forbids all loops. An implementation
> satisfying it generated zero loop motifs across the full sweep.

**Loop non-vacuity (guards against satisfying reducibility by generating
no loops at all):**
Over the same grid, assert every generated graph contains at least one
cycle. This holds by construction: `num_loops ≥ 1` attempts are made and
the first attempt always has at least the entry `ids[0]` as a proper
dominator candidate, so at least one back edge is added.

**Canonicality property (zero discards):**
For `num_nodes in {12, 20}` × `edge_prob = 0.2` × `seed in range(20)`:
encode via the full pipeline (reduce → extract → build → encode), rebuild
the same graph under a seed-derived permutation of node ids, re-encode, and
assert the token streams are EQUAL (hard assert — no skip fallback, unlike
the pre-existing round-trip test).

**Determinism retained:** same seed twice ⇒ identical edge set, at the new
RNG ordering (the existing determinism test already covers this; add one
case at `edge_prob=0.5` only if not already covered).

Reuse helper style from `tests/test_linearize.py` (`_graph_edges`,
seed-derived mapping); duplicating small helpers into `test_gen.py` is fine
and keeps `test_gen.py` free of cross-test-module imports.

## Acceptance (measured after implementation, not a unit test)

`scripts/gen_corpus.py --min-nodes 6 --max-nodes 20
--edge-probs 0.10,0.15,0.20,0.25,0.30 --seeds-per-config 25` should report
`discarded=0` (or near zero — any residual discard indicates a
label-dependence leak other than headers and must be reported, not hidden).

## Constraints

- Python 3.13+, stdlib only, existing code style
- `tests/test_linearize.py` and all other existing files unchanged
- Verify: `uv run python -m pytest tests/ -q` and `uv run ty check`
