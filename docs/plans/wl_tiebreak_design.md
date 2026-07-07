# WL-Signature Tie-Break for `canonical_order` — Design (deferred)

**Status: design only — implementation deferred.** The canonicality
self-check in `scripts/gen_corpus.py` already discards every affected
graph, so corpus correctness does not depend on this fix. Implementing it
only raises sweep yield from ~97.8% to (expected) 100%. Priority is below
PoC-0 training; pick this up when discard rate starts to matter (e.g.
large-scale corpus generation where 2% is real compute).

## Problem

`canonical_order()` (cfg_reducer/linearize.py) sorts each Kahn layer by

```python
(
    tuple(sorted(canonical_index[parent] for parent in parents[step])),
    _KIND_RANK[motifs_by_step[step].kind],
    motif_positions[step],   # ← label-dependent residual tie-break
)
```

When two motifs in the same layer have identical `(parent-index tuple,
kind)` but are **not automorphic** (their descendant structure differs),
the final key `motif_positions[step]` — the position in `mg.motifs`, i.e.
reduction step order, i.e. ultimately node-id order — decides their
relative position. A node-id relabeling can then swap them and produce a
different token stream.

Measured 2026-07-05: 42 / 1875 sweep graphs (2.2%) discarded for exactly
this reason. Repro: `n=7, p=0.1, seed=0` — layer key collision
`((0,), 'linear'): ['N03', 'N01']`; the two linear motifs have different
children, so the streams diverge at the later merge's ptr list.

## Fix: structural signature between kind-rank and position

Insert a **WL-style (Weisfeiler–Leman color refinement) signature** into
the sort key, so that ties are broken by structure before falling back to
position:

```python
(
    parent_indices,          # unchanged
    kind_rank,               # unchanged
    wl_signature[step],      # NEW — label-free structural fingerprint
    motif_positions[step],   # kept as ultimate fallback (see Limits)
)
```

### Signature computation (per level, bottom-up over subgraphs)

Operates on the same dedup'd adjacency as `canonical_order` (via
`_adjacency(mg)`), entirely label-free:

1. **Initial color** of each motif:
   - non-loop: `(kind,)`
   - loop: `(kind, skeleton_fingerprint(subgraph))` — a recursive,
     canonical hash of the loop's child level. Because signatures are
     computed bottom-up (innermost levels first), the child level's own
     ordering is already WL-stabilized when the parent needs it, so the
     fingerprint is well-defined. Reuse `skeleton_of()` on the child
     `MetaGraph` and hash the resulting `Skeleton` (it is
     hashable-by-value once converted to nested tuples).
2. **Refinement round**: new color of `m` =
   `hash(old_color, sorted multiset of in-neighbor old colors,
   sorted multiset of out-neighbor old colors)`.
   Direction matters — in- and out-neighborhoods are kept separate
   (a CFG metagraph is a DAG; symmetrizing would lose information).
3. Iterate until the color partition stabilizes (≤ `len(motifs)` rounds;
   levels are small, this is cheap).
4. `wl_signature[step]` = final color, made deterministic across runs by
   representing colors as canonical nested tuples (NOT Python `hash()`,
   which is salted for strings — use the tuple itself or
   `hashlib.sha256` over a canonical repr).

### Where it hooks in

Only `canonical_order()` changes: compute signatures once at entry, add
one key component. `encode` / `skeleton_of` / `decode` are untouched —
they consume `canonical_order`'s output. The token grammar is unchanged;
existing corpora remain valid (streams for already-canonical graphs must
not change — see Tests).

## Limits (why the position fallback stays)

1-WL is not a complete graph invariant: WL-equivalent but non-automorphic
motif pairs can theoretically survive refinement (CFG metagraph layers are
small DAGs, so this should be vanishingly rare, but it is not impossible).
The `motif_positions` fallback therefore remains as the last key, and the
**gen_corpus self-check filter must stay in place** as the correctness
backstop. Expected outcome: discard drops from 2.2% to 0 on the standard
sweep; any residual discard is a WL-indistinguishable pair and should be
reported with a repro, not hidden.

## Tests (TDD — write first)

1. **Repro goes green (hard assert)**: `n=7, p=0.1, seed=0` pipeline,
   seed-derived relabeling, token streams equal. Currently fails.
2. **Stream stability**: for graphs that are already canonical (the
   handcrafted diamond / loop / nested-loop cases with exact-stream
   assertions), `encode` output is byte-identical before and after the
   change — the WL key must only reorder previously-tied motifs.
3. **Loop-content sensitivity**: two loop motifs in one layer with
   identical (parents, kind) but different child skeletons must order by
   child structure, invariant under relabeling (handcraft: two sibling
   loops, one containing 2 nodes, one containing 3).
4. **Sweep acceptance** (not a unit test): standard sweep
   `--min-nodes 6 --max-nodes 20 --edge-probs 0.10..0.30
   --seeds-per-config 25` reports `discarded=0`.

## Constraints

- Python 3.13+, stdlib only, style of existing `linearize.py`.
- No changes to token grammar, `types.py`, `gen_corpus.py`.
- Verify: `uv run python -m pytest tests/ -q`, `uv run ty check`.
