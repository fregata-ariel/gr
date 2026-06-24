"""
Motif extraction — interpret Op history in reverse to recover
structural building blocks (Entry, Linear, Merge, Loop).

The reduction algorithm *removes* structure; replaying in reverse
*reconstructs* it.  Each undo step maps to exactly one Motif.

Loop Motifs are containers: child Motifs represent the internal
structure of the SCC, detected via scope tracking on the Op stream.
"""

from __future__ import annotations

from .types import Op, Motif


# ── single-op classifiers ────────────────────

def _classify_remove_node(op: Op) -> Motif:
    inv = op.inverse
    preds = tuple(inv["pred_edges"])
    succs = tuple(inv["succ_edges"])
    node = inv["target"]

    n_preds = len(preds)
    if n_preds == 0:
        kind = "entry"
    elif n_preds == 1:
        kind = "linear"
    else:
        kind = "merge"

    return Motif(kind=kind, node=node, preds=preds, succs=succs)


# ── scope tracking ───────────────────────────

def _build_parent_map(ops: list[Op]) -> dict[int, int]:
    """
    Walk ops in forward order and return {child_index: parent_index}.

    A `remove_edges` op activates a scope; subsequent `remove_node`
    and nested `remove_edges` ops within that scope are children.
    Scopes nest: an inner SCC (subset of outer) is tracked as a
    child of the outer `remove_edges`, and its members are removed
    from the outer scope's tracking set.
    """
    parent_map: dict[int, int] = {}
    scope_stack: list[tuple[int, set[str]]] = []

    for i, op in enumerate(ops):
        if op.kind == "remove_edges":
            scc = set(op.meta.get("scc", []))
            if scope_stack:
                parent_idx, parent_scc = scope_stack[-1]
                if scc <= parent_scc:
                    parent_map[i] = parent_idx
                    parent_scc -= scc
            scope_stack.append((i, scc))

        elif op.kind == "remove_node":
            target = op.forward["target"]
            if scope_stack:
                parent_idx, scc_nodes = scope_stack[-1]
                if target in scc_nodes:
                    parent_map[i] = parent_idx
                    scc_nodes.discard(target)
                    if not scc_nodes:
                        scope_stack.pop()

    return parent_map


def _compute_loop_interface(
    op: Op, child_ops: list[Op], all_ops: list[Op],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Compute the external preds/succs of a Loop Motif.

    preds: from child ops' pred_edges that reference nodes outside SCC.
    succs: scan all ops for remove_node targets outside the SCC whose
           pred_edges include an SCC member (these nodes were removed
           before the SCC scope, so the child ops can't see them).
    """
    scc = set(op.meta.get("scc", []))

    external_preds: set[str] = set()
    for child_op in child_ops:
        if child_op.kind != "remove_node":
            continue
        for p in child_op.inverse["pred_edges"]:
            if p not in scc:
                external_preds.add(p)

    external_succs: set[str] = set()
    for other_op in all_ops:
        if other_op.kind != "remove_node":
            continue
        target = other_op.forward["target"]
        if target in scc:
            continue
        if any(p in scc for p in other_op.inverse["pred_edges"]):
            external_succs.add(target)

    return tuple(sorted(external_preds)), tuple(sorted(external_succs))


# ── main extraction ──────────────────────────

def extract(ops: list[Op]) -> list[Motif]:
    """
    Walk the Op history and return Motifs in reconstruction
    (generation) order as a forest with Loop containers.

    Returns only top-level Motifs; Loop children are nested inside
    their parent's ``children`` field.
    """
    parent_map = _build_parent_map(ops)

    children_of: dict[int, list[int]] = {}
    for child_idx, parent_idx in parent_map.items():
        children_of.setdefault(parent_idx, []).append(child_idx)

    top_level_indices = [
        i for i in range(len(ops)) if i not in parent_map
    ]

    step_counter = 0

    def _build_motif(idx: int) -> Motif:
        nonlocal step_counter
        op = ops[idx]

        if op.kind == "remove_node":
            m = _classify_remove_node(op)
            s = step_counter
            step_counter += 1
            return Motif(kind=m.kind, node=m.node, preds=m.preds,
                         succs=m.succs, meta=m.meta, step=s)

        if op.kind == "remove_edges":
            child_indices = children_of.get(idx, [])
            child_ops = [ops[ci] for ci in child_indices]

            child_motifs = []
            for ci in reversed(child_indices):
                child_motifs.append(_build_motif(ci))

            ext_preds, ext_succs = _compute_loop_interface(op, child_ops, ops)

            edges = op.inverse.get("edges", op.forward.get("edges", []))
            back_edges = [tuple(e) for e in edges]
            header = op.meta.get("header")
            scc = op.meta.get("scc", [])

            s = step_counter
            step_counter += 1
            return Motif(
                kind="loop",
                node=None,
                preds=ext_preds,
                succs=ext_succs,
                meta={"header": header, "scc": scc, "back_edges": back_edges},
                step=s,
                children=tuple(child_motifs),
            )

        step_counter += 1
        return Motif(kind=op.kind, node=None, preds=(), succs=(),
                     step=step_counter - 1)

    motifs: list[Motif] = []
    for idx in reversed(top_level_indices):
        motifs.append(_build_motif(idx))

    return motifs
