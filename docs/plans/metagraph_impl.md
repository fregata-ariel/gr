# MetaGraph Implementation Plan

## Goal

Add a `MetaGraph` data type and `build()` function that converts a hierarchical
Motif tree (produced by `motif.extract()`) into a DAG representing structural
dependencies between Motifs.

## Files to modify/create

| File | Action | What |
|------|--------|------|
| `cfg_reducer/types.py` | Add | `MetaGraph` frozen dataclass |
| `cfg_reducer/metagraph.py` | Create | `build(motifs) -> MetaGraph` |
| `cfg_reducer/__init__.py` | Update | Export `MetaGraph`, `metagraph` |
| `tests/test_metagraph.py` | Create | 3 test cases |

## 1. MetaGraph dataclass (`cfg_reducer/types.py`)

Append after the `Motif` class:

```python
@dataclass(frozen=True)
class MetaGraph:
    """
    DAG of Motif dependencies at one level of the hierarchy.

    motifs:     Motif nodes at this level (ordered by step).
    edges:      (src_step, dst_step) pairs — src restored a node that
                dst references in preds/succs.
    subgraphs:  Loop Motif step → MetaGraph of its children.
    """
    motifs: tuple[Motif, ...]
    edges: tuple[tuple[int, int], ...]
    subgraphs: dict[int, 'MetaGraph'] = field(default_factory=dict)
```

Keep the existing import of `field` from `dataclasses`.
Use `tuple` (not `list`) for `motifs` and `edges` to match the frozen convention.
`subgraphs` uses `dict` — same pattern as `Motif.meta`.

## 2. `build()` function (`cfg_reducer/metagraph.py`)

```
cfg_reducer/metagraph.py
```

Module-level docstring: one-liner explaining the transformation.

### Algorithm

```python
def build(motifs: list[Motif]) -> MetaGraph:
```

**Step 1 — Node-to-step lookup** for this level only:

```
node_to_step: dict[str, int] = {}
for m in motifs:
    if m.kind == "loop":
        for n in m.meta["scc"]:
            node_to_step[n] = m.step
    elif m.node is not None:
        node_to_step[m.node] = m.step
```

- Non-loop Motifs register their single `node`.
- Loop Motifs register every SCC member (the Loop represents the entire SCC
  as one node in the metagraph).

**Step 2 — Edge construction**:

```
edge_set: set[tuple[int, int]] = set()
for m in motifs:
    for dep_node in (*m.preds, *m.succs):
        if dep_node in node_to_step:
            src_step = node_to_step[dep_node]
            if src_step != m.step:
                edge_set.add((src_step, m.step))
```

- For each Motif, scan its `preds` and `succs`.
- If a referenced node is restored by another Motif at this level → add edge.
- Skip self-edges (should not occur in practice but guard anyway).
- Nodes not in `node_to_step` are outside this level (cross-level refs in
  Loop subgraphs) — they are correctly ignored.

**Step 3 — Recursive subgraphs**:

```
subgraphs: dict[int, MetaGraph] = {}
for m in motifs:
    if m.kind == "loop" and m.children:
        subgraphs[m.step] = build(list(m.children))
```

**Return**:

```
return MetaGraph(
    motifs=tuple(motifs),
    edges=tuple(sorted(edge_set)),
    subgraphs=subgraphs,
)
```

Sort edges for deterministic output.

### Imports

```python
from .types import Motif, MetaGraph
```

No other imports needed.

## 3. `__init__.py` update

Add `MetaGraph` to the import from `.types` and add `metagraph` to the
module import. Add both to `__all__`.

Current:
```python
from .types import NodeType, Op, Motif, BASIC
from .engine import GraphEngine, Node
from .algorithm import ReductionAlgorithm, Scope, tarjan_scc
from . import store, motif
```

Target:
```python
from .types import NodeType, Op, Motif, MetaGraph, BASIC
from .engine import GraphEngine, Node
from .algorithm import ReductionAlgorithm, Scope, tarjan_scc
from . import store, motif, metagraph
```

And add `"MetaGraph"` and `"metagraph"` to `__all__`.

## 4. Tests (`tests/test_metagraph.py`)

Use the existing reduction pipeline to generate Ops, then extract Motifs,
then build MetaGraph. Each test should:

1. Build a graph with `GraphEngine`
2. Run `ReductionAlgorithm` to completion
3. Call `motif.extract(engine.history)` to get Motifs
4. Call `metagraph.build(motifs)` to get MetaGraph
5. Assert on the structure

### Test A: Diamond (no loops)

Graph: A→B, A→C, B→D, C→D

Expected MetaGraph (top level, no subgraphs):
- 4 motifs: Entry(A), Linear(B), Linear(C), Merge(D)
- Edges: Entry(A)→Linear(B), Entry(A)→Linear(C), Linear(B)→Merge(D), Linear(C)→Merge(D)
- `subgraphs` is empty

Assertions:
```python
assert len(mg.motifs) == 4
assert len(mg.edges) == 4
assert mg.subgraphs == {}
# Verify edge connectivity by checking step pairs
```

### Test B: Simple loop

Graph: A→B, B→C, C→B, C→D

Expected MetaGraph (top level):
- 3 motifs: Entry(A), Loop({B,C}), Linear(D)
- Edges: Entry(A)→Loop, Loop→Linear(D)
- `subgraphs` has 1 entry (the Loop's step)

Loop subgraph:
- 2 motifs: some ordering of B, C
- 1 edge between them
- No further subgraphs

Assertions:
```python
assert len(mg.motifs) == 3
assert len(mg.edges) == 2
assert len(mg.subgraphs) == 1
sub = list(mg.subgraphs.values())[0]
assert len(sub.motifs) == 2
assert len(sub.edges) == 1
```

### Test C: Nested loop

Graph: A→B, B→C, C→D, D→C, D→B, B→E

Expected MetaGraph (top level):
- 3 motifs: Entry(A), Loop({B,C,D}), Linear(E)
- 2 edges: Entry(A)→Loop, Loop→Linear(E)
- 1 subgraph (outer Loop)

Outer Loop subgraph:
- Contains Entry(B) and nested Loop({C,D})
- 1 edge: Entry(B)→inner Loop
- 1 subgraph (inner Loop)

Inner Loop subgraph:
- 2 motifs for C, D
- 1 edge between them

Assertions:
```python
assert len(mg.motifs) == 3
assert len(mg.subgraphs) == 1
outer_sub = list(mg.subgraphs.values())[0]
assert len(outer_sub.subgraphs) == 1
inner_sub = list(outer_sub.subgraphs.values())[0]
assert len(inner_sub.motifs) == 2
assert inner_sub.subgraphs == {}
```

## Constraints

- Python 3.13+
- No new dependencies
- Follow existing code style in `cfg_reducer/` (type hints, minimal docstrings)
- `MetaGraph` must be frozen dataclass (consistent with `Motif`)
- Use `from __future__ import annotations` in metagraph.py (consistent with motif.py)
- Run `uv run python -m pytest tests/test_metagraph.py` to verify
