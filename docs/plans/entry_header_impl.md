# Entry-Aware Header Selection Implementation Plan

## Goal

Treat a CFG as **(G, entry)** rather than a bare digraph. When a terminal
SCC contains the designated entry node (a "whole-program loop", common now
that dominator-gated back edges frequently target `ids[0]`), such an SCC
has no external predecessors and `_identify_header()` currently falls back
to `min(scc)` — a node-id-dependent choice that breaks canonicality.
Measured 2026-07-05: 41% of the full sweep is discarded by the canonicality
self-check, almost entirely from this case.

The semantic header of an entry-containing loop is the entry itself.
Passing the entry through the pipeline makes header selection canonical
under entry-preserving relabelings.

Key structural fact (why this is a clean three-way fallback): if
`entry ∈ SCC`, the SCC cannot have external predecessors — any external
node with an edge into the SCC would itself be reachable from the entry
(inside the SCC), closing a cycle that pulls it into the SCC. So
`external_entries` and `entry ∈ scc` are mutually exclusive cases and
existing behavior for non-entry SCCs is unchanged.

## Files to modify

| File | Action | What |
|------|--------|------|
| `cfg_reducer/algorithm.py` | Update | `entry` parameter + header fallback |
| `scripts/gen_corpus.py` | Update | thread entry through pipeline + self-check |
| `tests/test_gen.py` | Extend/Update | entry threading in canonicality test |
| `tests/test_linearize.py` | Update | helper accepts optional entry; entry-SCC case |

`types.py`, `engine.py`, `motif.py`, `metagraph.py`, `linearize.py`,
`gen.py`, `main.py` are untouched (`main.py`'s visualizer keeps the
no-entry default).

## 1. `cfg_reducer/algorithm.py`

```python
def __init__(self, engine: GraphEngine, entry: str | None = None) -> None:
    ...
    self.entry = entry
```

`_identify_header()` becomes a three-way fallback:

```python
if external_entries:
    return min(external_entries)
if self.entry is not None and self.entry in scc:
    return self.entry
return min(scc)
```

No other behavior changes. Ops, undo/redo, store serialization are
unaffected (`entry` is a runtime parameter, not recorded state).

## 2. `scripts/gen_corpus.py`

- `_extract_metagraph(engine, entry)` passes entry to `ReductionAlgorithm`.
- `_build_record` passes `"N00"` (the `build_cfg` convention: entry is
  `ids[0]`).
- `_passes_canonicality_self_check` passes `mapping["N00"]` for the
  permuted graph — relabelings must map the entry designation along with
  the node ids.

## 3. Tests (written FIRST — TDD)

**`tests/test_linearize.py`:**
- `_extract_metagraph(edges, entry=None)` — optional entry threaded to
  `ReductionAlgorithm`; existing handcrafted-graph tests keep passing
  without entry (their loops have external preds; behavior identical).
- New handcrafted entry-SCC case: `A→B, B→C, C→A, C→D` (SCC {A,B,C}
  contains the entry, no external preds):
  - with `entry="A"`: the loop motif's `meta["header"] == "A"` and
    `meta["back_edges"] == [("C", "A")]`
  - canonicality: relabel with a permutation, pass the mapped entry,
    assert identical token streams (hard assert)
  - without entry: header falls back to `min(scc)` (documents the
    default-compatible behavior)
- `test_decode_encode_round_trip_for_build_cfg_seeds`: pass entry
  (`"N00"` / mapped) through both pipelines; keep the skip guard —
  after this change it should never trigger (zero skips expected).

**`tests/test_gen.py`:**
- `test_build_cfg_is_canonical_under_seed_derived_node_id_permutations`:
  thread entry through both encodes (`"N00"` original, `mapping["N00"]`
  relabeled). This is the currently-failing test; with entry support it
  must go green.

**Red state before implementation:** tests passing `entry=...` to
`ReductionAlgorithm` fail with `TypeError` (parameter does not exist yet);
the gen canonicality property test keeps failing on entry-SCC graphs.

## Acceptance (after implementation)

- Full suite green, **zero skips**.
- Sweep `--min-nodes 6 --max-nodes 20 --edge-probs 0.10,...,0.30
  --seeds-per-config 25` reports **discarded=0** (any residual discard is
  a further leak — report, do not hide).

## Constraints

- Python 3.13+, stdlib only, existing code style
- Verify: `uv run python -m pytest tests/ -q` and `uv run ty check`
  (sandbox fallback: `./.venv/bin/python -m pytest`, `./.venv/bin/ty`)
