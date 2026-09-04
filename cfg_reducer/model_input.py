"""
Model-input transformation — flatten canonical MetaGraphs and tokenize
them for the hierarchical autoregressive baseline.

One-way, topology-only views derived from the canonical structure:
steps and node ids stay in the canonical schema (node-token assignment
is a later phase).  Design: docs/design/model_input.md.
"""

from __future__ import annotations

from .types import MetaGraph

KIND_TOKENS = {
    "entry": "KIND_ENTRY",
    "linear": "KIND_LINEAR",
    "merge": "KIND_MERGE",
    "loop": "KIND_LOOP",
}

_SPECIAL = ("PAD", "BOS", "EOS", "LOOP_START", "LOOP_END")

# Positional topology sketch of one level:
#   ((kind, in_offsets, child_sketch_or_None), ...)
Sketch = tuple


def _level_in_offsets(mg: MetaGraph) -> list[tuple]:
    """Per-motif (motif, sorted backward in-edge offsets) for one level."""
    level = sorted(mg.motifs, key=lambda m: m.step)
    pos_of = {m.step: i for i, m in enumerate(level)}

    incoming: dict[int, list[int]] = {m.step: [] for m in level}
    for src, dst in mg.edges:
        incoming[dst].append(pos_of[dst] - pos_of[src])

    return [(m, sorted(incoming[m.step])) for m in level]


def flatten(mg: MetaGraph) -> list[dict]:
    """
    Flat table in DFS pre-order (a Loop's children follow its row).

    Rows keep `step` as the mapping back to the canonical sample and
    add the derived `parent_loop_step` / `depth` fields (A3-2).
    """
    rows: list[dict] = []

    def walk(g: MetaGraph, parent_loop_step: int | None, depth: int) -> None:
        for position, (m, offsets) in enumerate(_level_in_offsets(g)):
            rows.append({
                "step": m.step,
                "kind": m.kind,
                "parent_loop_step": parent_loop_step,
                "depth": depth,
                "position": position,
                "in_offsets": offsets,
            })
            if m.kind == "loop" and m.step in g.subgraphs:
                walk(g.subgraphs[m.step], m.step, depth + 1)

    walk(mg, None, 0)
    return rows


def sketch_of(mg: MetaGraph) -> Sketch:
    """Positional topology of a level: what tokenize/detokenize preserve."""
    items = []
    for m, offsets in _level_in_offsets(mg):
        child = None
        if m.kind == "loop":
            sub = mg.subgraphs.get(m.step)
            child = sketch_of(sub) if sub is not None else ()
        items.append((m.kind, tuple(offsets), child))
    return tuple(items)


def max_offset_needed(mg: MetaGraph) -> int:
    """Largest backward offset in the sample (for sizing the vocab)."""
    largest = 0
    for _, offsets in _level_in_offsets(mg):
        if offsets:
            largest = max(largest, offsets[-1])
    for sub in mg.subgraphs.values():
        largest = max(largest, max_offset_needed(sub))
    return largest


def build_vocab(max_offset: int) -> dict[str, int]:
    """Deterministic token -> id table (PAD is always 0)."""
    if max_offset < 1:
        raise ValueError(f"max_offset must be >= 1, got {max_offset}")
    tokens = [
        *_SPECIAL,
        *(KIND_TOKENS[k] for k in ("entry", "linear", "merge", "loop")),
        *(f"REF_{k}" for k in range(1, max_offset + 1)),
    ]
    return {token: i for i, token in enumerate(tokens)}


def tokenize(mg: MetaGraph, vocab: dict[str, int]) -> list[int]:
    """MetaGraph -> token ids per the grammar in docs/design/model_input.md."""
    out = [vocab["BOS"]]

    def emit_level(g: MetaGraph) -> None:
        for m, offsets in _level_in_offsets(g):
            kind_token = KIND_TOKENS.get(m.kind)
            if kind_token is None:
                raise ValueError(
                    f"unknown motif kind {m.kind!r} at step {m.step}: "
                    f"canonical vocabulary is {sorted(KIND_TOKENS)}"
                )
            out.append(vocab[kind_token])

            for offset in offsets:
                ref = f"REF_{offset}"
                if ref not in vocab:
                    raise ValueError(
                        f"offset {offset} at step {m.step} exceeds the "
                        "vocab window; rebuild with a larger max_offset"
                    )
                out.append(vocab[ref])

            if m.kind == "loop":
                out.append(vocab["LOOP_START"])
                sub = g.subgraphs.get(m.step)
                if sub is not None:
                    emit_level(sub)
                out.append(vocab["LOOP_END"])

    emit_level(mg)
    out.append(vocab["EOS"])
    return out


def detokenize(token_ids: list[int], vocab: dict[str, int]) -> Sketch:
    """
    Strictly parse a token stream back into a Sketch.

    Raises ValueError on any grammar violation, so this doubles as the
    well-formedness check for model-generated streams.
    """
    names = {i: t for t, i in vocab.items()}
    try:
        tokens = [names[i] for i in token_ids]
    except KeyError as exc:
        raise ValueError(f"unknown token id {exc.args[0]}") from None

    if len(tokens) < 2 or tokens[0] != "BOS" or tokens[-1] != "EOS":
        raise ValueError("stream must be BOS ... EOS")

    # Mutable levels: list of [kind, [offsets], child_level_or_None].
    # Two flag stacks mirror `stack`:
    #   refs_open    — REF is legal only directly after the level's
    #                  latest KIND (or one of its REFs)
    #   pending_loop — the latest KIND was a loop whose LOOP_START has
    #                  not arrived yet (its bracket pair is mandatory)
    root: list[list] = []
    stack = [root]
    refs_open = [False]
    pending_loop = [False]

    for token in tokens[1:-1]:
        level = stack[-1]

        if token in ("BOS", "EOS", "PAD"):
            raise ValueError(f"unexpected {token} inside the stream")

        if token.startswith("KIND_"):
            if pending_loop[-1]:
                raise ValueError("loop motif is missing its LOOP_START")
            kind = token.removeprefix("KIND_").lower()
            level.append([kind, [], [] if kind == "loop" else None])
            refs_open[-1] = True
            pending_loop[-1] = kind == "loop"

        elif token.startswith("REF_"):
            if not refs_open[-1]:
                raise ValueError("REF token with no preceding motif")
            offsets = level[-1][1]
            offset = int(token.removeprefix("REF_"))
            if offsets and offset <= offsets[-1]:
                raise ValueError("REF offsets must be strictly increasing")
            if offset > len(level) - 1:
                raise ValueError(
                    f"REF_{offset} points before the start of its level"
                )
            offsets.append(offset)

        elif token == "LOOP_START":
            if not pending_loop[-1]:
                raise ValueError("LOOP_START must follow a loop motif")
            refs_open[-1] = False
            pending_loop[-1] = False
            stack.append(level[-1][2])
            refs_open.append(False)
            pending_loop.append(False)

        elif token == "LOOP_END":
            if len(stack) < 2:
                raise ValueError("unbalanced LOOP_END")
            if pending_loop[-1]:
                raise ValueError("loop motif is missing its LOOP_START")
            stack.pop()
            refs_open.pop()
            pending_loop.pop()
            refs_open[-1] = False

        else:   # pragma: no cover — vocab only contains the tokens above
            raise ValueError(f"unhandled token {token}")

    if len(stack) != 1:
        raise ValueError("unbalanced LOOP_START")
    if pending_loop[-1]:
        raise ValueError("loop motif is missing its LOOP_START")

    def freeze(level: list[list]) -> Sketch:
        return tuple(
            (kind, tuple(offsets), freeze(child) if child is not None else None)
            for kind, offsets, child in level
        )

    return freeze(root)
