# gr — CFG Reducer

A stepwise Control Flow Graph reduction engine with full undo/redo history.
The reduction trace is reinterpreted as structural building blocks (Motifs)
for downstream CFG generation via Graph Transformers.

## Architecture

```
cfg_reducer/
  types.py       — Pure data: NodeType, Op, Motif, MetaGraph
  engine.py      — GraphEngine: node/edge mutation with Op-based undo/redo
  algorithm.py   — ReductionAlgorithm: two-phase (terminal removal + SCC cycle breaking)
                   Scope enter/leave, tarjan_scc()
  motif.py       — Motif extraction from Op history (reverse replay)
                   Hierarchical: Loop Motifs are containers with children
  metagraph.py   — Hierarchical Motif trees to dependency DAGs
  store.py       — JSON serialization/deserialization of Op history and
                   MetaGraph samples (docs/design/metagraph_schema.md)
  generate.py    — Pure synthetic CFG generator (no matplotlib)
  dataset.py     — Batch dataset builder + CLI: splits, iso-dedup, manifest
  model_input.py — Flatten + tokenize for the AR baseline (topology-only view)
  __init__.py    — Public API re-exports
main.py          — Interactive matplotlib visualizer (imports generate_cfg)
training/        — AR baseline: local tokenize/eval (cfg_reducer, no torch),
                   Colab trainer train_ar.py (torch only, single file)
docs/            — Discussion logs and design notes
tests/           — Regression tests for engine, algorithm, motif, metagraph, store
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

## Handoff

Read `docs/handoff.md` for the current branch state, required reading order,
MetaGraph invariants, verification commands, and open design decisions.

## Companion Project

`/home/user/Projects/Compiler/pyClangAST/` — C/C++ AST parser (`calisp`) for building
real CFG training corpora. Read-only reference; not modified from this project.

## Delegation

Rules that delegated agents must follow are in `AGENTS.md` (both Codex and OpenCode
read it at start-up). Division of labour since 2026-09-09: Claude Code orchestrates and
takes design decisions; Codex (`gpt-6-astra`, effort `high`) writes detailed designs;
Codex (`gpt-6-astra`, effort `low`) implements one or two design tasks per run; OpenCode
(`opencode-go/deepseek-v4.1-flash`, `opencode-go/muse-spark-1.3-contributor`) is used
for prototype-level scripts and bounded edits. Every delegated result is re-verified
here (`uv run pytest -q`, `uv run ty check`, `git diff --check`) before it is committed.

- Codex: `node ~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs
  task --background --write --model gpt-6-astra --effort <high|low> "$(cat prompt.md)"`,
  then `status` / `result`. Write the prompt to a file with a quoted heredoc; a bash
  double-quoted string swallows backticks.
- OpenCode: run in a data-free `git worktree add --detach /tmp/oc-wt HEAD` (the real
  checkout's `data/` and `runs/` make its init hang) and always under `timeout`.
  Read-only generation: `opencode run --pure --agent plan -m <model> --dir /tmp/oc-wt --
  "<prompt>"`; bounded edits: `--agent build --auto` in the worktree, then review the
  diff. Never put backticks in the message or an attached file — `opencode run` stalls
  silently on them. A run that produces no output within a few minutes is a stall:
  retry once.
