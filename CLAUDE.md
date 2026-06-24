# gr — CFG Reducer

A stepwise Control Flow Graph reduction engine with full undo/redo history.
The reduction trace is reinterpreted as structural building blocks (Motifs)
for downstream CFG generation via Graph Transformers.

## Architecture

```
cfg_reducer/
  types.py       — Pure data: NodeType, Op, Motif (frozen dataclasses)
  engine.py      — GraphEngine: node/edge mutation with Op-based undo/redo
  algorithm.py   — ReductionAlgorithm: two-phase (terminal removal + SCC cycle breaking)
                   Scope enter/leave, tarjan_scc()
  motif.py       — Motif extraction from Op history (reverse replay)
                   Hierarchical: Loop Motifs are containers with children
  store.py       — JSON serialization/deserialization of Op history
  __init__.py    — Public API re-exports
main.py          — Interactive matplotlib visualizer with build_cfg()
docs/            — Discussion logs and design notes
```

## Key Concepts

- **Op history**: Every graph mutation is recorded as an Op with forward/inverse params and metadata.
  Ops are the single source of truth for both undo/redo and Motif extraction.
- **Motif kinds**: entry (no preds), linear (1 pred), merge (2+ preds), loop (SCC container).
  Loop Motifs hold child Motifs and expose external preds/succs as their interface.
- **Metagraph**: Motifs form a DAG where edges represent node-sharing dependencies —
  if Motif M_i's restored node appears in M_j's preds/succs, then M_i → M_j.
  Loop Motifs represent their entire SCC as a single node in the metagraph.

## Conventions

- Python 3.13+, managed with `uv`.
- Data types are frozen dataclasses in types.py — keep them serialization-friendly.
- No runtime dependencies beyond matplotlib and networkx.
- Type checker: `ty` (in dev dependencies).

## Companion Project

`/home/user/Projects/Compiler/pyClangAST/` — C/C++ AST parser (`calisp`) for building
real CFG training corpora. Read-only reference; not modified from this project.
