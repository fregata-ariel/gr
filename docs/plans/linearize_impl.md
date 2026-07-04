# Linearize + Corpus Generation Implementation Plan (PoC-0)

## Goal

Add the final pipeline stage for PoC-0 training data:

1. `cfg_reducer/gen.py` — move `build_cfg()` out of `main.py` (no behavior change)
2. `cfg_reducer/linearize.py` — canonical serialization: `MetaGraph` → token
   sequence (`encode`), token sequence → structural skeleton (`decode`)
3. `scripts/gen_corpus.py` — parameter sweep, canonicality self-check,
   dedup, JSONL output

Development follows TDD: tests are written and reviewed first, then the
implementation.

## Files to modify/create

| File | Action | What |
|------|--------|------|
| `cfg_reducer/gen.py` | Create | `build_cfg()` moved from `main.py` verbatim |
| `main.py` | Update | import `build_cfg` from `cfg_reducer.gen`; keep re-export for callers |
| `cfg_reducer/types.py` | Add | `Skeleton` frozen dataclass |
| `cfg_reducer/linearize.py` | Create | `encode()`, `decode()`, `skeleton_of()`, `canonical_order()` |
| `cfg_reducer/__init__.py` | Update | export `gen`, `linearize`, `Skeleton` |
| `scripts/gen_corpus.py` | Create | corpus generation CLI |
| `tests/test_gen.py` | Create | build_cfg relocation tests |
| `tests/test_linearize.py` | Create | grammar / canonicality / round-trip tests |

## 1. `cfg_reducer/gen.py`

Move `build_cfg()` from `main.py` **verbatim** (same signature, same RNG usage,
same seeded determinism — an identical seed must produce the identical graph
before and after the move). Only stdlib `random` is imported; no matplotlib
or networkx.

`main.py` replaces its local definition with
`from cfg_reducer.gen import build_cfg` so existing behavior
(visualizer, tests using `main.build_cfg`) is preserved.

## 2. Token grammar

A `MetaGraph` (one hierarchy level) serializes to a flat token stream.
Tokens are plain strings.

```
level    := item* STOP
item     := ADD_ENTRY parents
          | ADD_LINEAR parents
          | ADD_MERGE  parents
          | ADD_LOOP   parents OPEN level CLOSE
parents  := ptr_k*          # ascending, no duplicates
```

- Each `item` receives a **level-local index** (0-based, in emission order).
  `ptr_k` refers to the k-th item of the **current level only** — pointers
  never cross levels (guaranteed by `metagraph.build()`: child-level edges
  are closed within the level).
- `parents` of an item are its in-neighbors in this level's dedup'd
  metagraph edge set (both pred-derived and succ-derived edges are treated
  uniformly as dependencies).
- Motif kind is encoded **explicitly** in the ADD token. It is NOT derivable
  from the level-local parent count in general: e.g. a merge whose CFG preds
  all lie in one SCC has a single metagraph parent (the Loop); a loop child
  whose CFG pred is outside the SCC loses that parent at the child level.
- Every level ends with an explicit `STOP` (uniform end-of-level decision
  point, including child levels — `CLOSE` follows the child `STOP`).

Example — simple loop `A→B, B→C, C→B, C→D`:

```
ADD_ENTRY                     # idx 0 (A)
ADD_LOOP ptr_0 OPEN           # idx 1 (SCC {B,C}); external pred = idx 0
    <child level items>       # child-local indices 0..
    STOP
CLOSE
ADD_LINEAR ptr_1              # idx 2 (D); parent = the loop
STOP
```

The child-level content depends on how in-loop motifs classify after
back-edge cutting; tests must derive expectations by running the real
pipeline, not by copying idealized examples from docs.

## 3. Canonical ordering

`canonical_order(mg: MetaGraph) -> list[int]` returns motif `step` ids in
emission order. `step` is used as an **opaque identifier only** — it must
never influence ordering (this implements the "step is never exposed"
decision from `docs/discussion_log.md` Session 4).

Algorithm (per level):

1. Kahn/BFS layering over the level's dedup'd edges: layer 0 = in-degree 0
   motifs; layer k = motifs whose parents all lie in layers < k.
2. Within a layer, sort by the key
   `(sorted list of parent canonical indices, kind_rank)` where
   `kind_rank = {entry: 0, linear: 1, merge: 2, loop: 3}`.
   Parents are already emitted, so their canonical indices are defined.
3. Assign canonical indices in emission order; recurse into loop subgraphs.

Motifs tied on the full key are structurally interchangeable at this level
(automorphic siblings, e.g. the two branches of a diamond); any stable
order among them is acceptable — use their relative position in
`mg.motifs` as the final tie-break so the result is deterministic.

`encode(mg: MetaGraph) -> list[str]` emits tokens in canonical order.
Parents are emitted as ascending `ptr_k` tokens.

## 4. `Skeleton` and `decode()`

Token streams cannot recover CFG node ids, so `decode()` returns a
structure-only skeleton. Add to `types.py` (serialization-friendly,
frozen — matches convention):

```python
@dataclass(frozen=True)
class Skeleton:
    """Structure-only view of one MetaGraph level: (kind, parent idx set)
    per motif in canonical order, plus nested loop skeletons."""
    items: tuple[tuple[str, tuple[int, ...]], ...]   # (kind, sorted parents)
    subgraphs: dict[int, 'Skeleton'] = field(default_factory=dict)
```

- `skeleton_of(mg: MetaGraph) -> Skeleton` — canonical skeleton straight
  from a MetaGraph (shares `canonical_order`).
- `decode(tokens: list[str]) -> Skeleton` — parses a token stream.
  Raises `ValueError` (with a message identifying the offending position)
  on any violation:
  - unknown token
  - `ptr_k` with `k >= number of items emitted so far at the current level`
    (forward/self reference)
  - non-ascending or duplicate `ptr_k` within one parents list
  - `OPEN` not immediately after an `ADD_LOOP` parents list / unbalanced
    `OPEN`/`CLOSE`
  - `CLOSE` without preceding child-level `STOP`
  - tokens after the top-level `STOP`, or stream ending without it

Round-trip contract: `decode(encode(mg)) == skeleton_of(mg)` for every
MetaGraph produced by the real pipeline.

## 5. `scripts/gen_corpus.py`

CLI (argparse), pure stdlib:

```
uv run python scripts/gen_corpus.py \
    --out corpus.jsonl \
    --min-nodes 6 --max-nodes 20 \
    --edge-probs 0.10,0.15,0.20,0.25,0.30 \
    --seeds-per-config 100
```

Per (num_nodes, edge_prob, seed) configuration:

1. `build_cfg` → run `ReductionAlgorithm` to completion →
   `motif.extract(engine.history)` → `metagraph.build` → `encode`
2. **Canonicality self-check**: permute the node id strings with a
   seed-derived shuffle, rebuild the same graph under permuted ids, rerun
   the pipeline, `encode` again. If the two token streams differ, the
   graph's canonical form is label-dependent (e.g. multi-entry loop hit the
   shelved header tie-break) → **discard and count**. This filters every
   canonicality violation without needing irreducibility detection.
3. Dedup by the joined token string (exact match = isomorphic metagraph,
   by canonicality).

JSONL record:

```json
{"id": "n12_p0.18_s42",
 "gen": {"num_nodes": 12, "edge_prob": 0.18, "seed": 42},
 "tokens": ["ADD_ENTRY", "ADD_LOOP", "ptr_0", "OPEN", ...],
 "stats": {"n_motifs": 9,
           "kinds": {"entry": 1, "linear": 5, "merge": 2, "loop": 1},
           "n_tokens": 23, "depth": 4, "max_width": 3, "loop_nest": 1}}
```

- `n_motifs`, `kinds` — totals across all levels
- `depth`, `max_width` — Kahn layer count / max layer size at top level
- `loop_nest` — max loop nesting depth (0 = no loops)

Final line on stderr: totals — generated, discarded (canonicality),
deduped, written.

## 6. Tests (written FIRST — TDD)

Test files use the real pipeline as in `tests/test_metagraph.py`:
build graph → reduce to completion → `motif.extract` → `metagraph.build`.
Where exact token streams are asserted, they must be hand-derivable;
otherwise assert structural properties.

### `tests/test_gen.py`

- `build_cfg` importable from `cfg_reducer.gen`; `main.build_cfg` still works
- Same seed → identical edge set across two calls (determinism)
- Returned engine node count == `num_nodes`

### `tests/test_linearize.py`

**Exact-stream case (hand-verifiable):**
- Diamond `A→B, A→C, B→D, C→D`:
  `[ADD_ENTRY, ADD_LINEAR, ptr_0, ADD_LINEAR, ptr_0, ADD_MERGE, ptr_1, ptr_2, STOP]`

**Structural cases (via pipeline, no exact child-level streams):**
- Simple loop `A→B, B→C, C→B, C→D`: top level is
  `ADD_ENTRY`, `ADD_LOOP ptr_0 OPEN … STOP CLOSE`, `ADD_LINEAR ptr_1`, `STOP`;
  exactly one OPEN/CLOSE pair; child level contains exactly 2 ADD tokens
- Nested loop `A→B, B→C, C→D, D→C, D→B, B→E`: OPEN/CLOSE nesting depth 2

**Canonicality (hard assert, reducible graphs only):**
- For diamond / simple loop / nested loop: rebuild with permuted node ids
  (e.g. reversed name assignment), rerun pipeline → identical token stream

**Round-trip / decode:**
- `decode(encode(mg)) == skeleton_of(mg)` for the three handcrafted graphs
  and for `build_cfg` seeds 0–9 (skip any seed the canonicality self-check
  would discard, mirroring gen_corpus)
- `decode` raises `ValueError` for: forward `ptr` reference, duplicate ptr,
  unbalanced OPEN/CLOSE, missing top-level STOP, trailing tokens after STOP,
  unknown token

**canonical_order:**
- Output is a permutation of the level's steps
- Never inspects `step` for ordering: two MetaGraphs identical up to a
  relabeling of `step` values yield the same token stream (construct by
  building the same graph twice — steps are equal — then manually remapping
  `step` values in a copied MetaGraph)

## Constraints

- Python 3.13+, no new dependencies (stdlib only for all new modules)
- Frozen dataclasses in `types.py`; keep serialization-friendly
- `from __future__ import annotations` in new modules (matches motif.py)
- Follow existing code style (type hints, minimal docstrings)
- Verify: `uv run python -m pytest tests/ -q` and `uv run ty check`
