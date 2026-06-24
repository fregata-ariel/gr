"""
Motif extraction — interpret Op history in reverse to recover
structural building blocks (Entry, Linear, Merge, Loop).

The reduction algorithm *removes* structure; replaying in reverse
*reconstructs* it.  Each undo step maps to exactly one Motif.
"""

from __future__ import annotations

from .types import Op, Motif


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


def _classify_remove_edges(op: Op) -> Motif:
    edges = op.inverse.get("edges", op.forward.get("edges", []))
    back_edges = [tuple(e) for e in edges]
    header = op.meta.get("header")
    scc = op.meta.get("scc", [])

    return Motif(
        kind="loop",
        node=None,
        preds=(),
        succs=(),
        meta={"header": header, "scc": scc, "back_edges": back_edges},
    )


_CLASSIFIERS = {
    "remove_node": _classify_remove_node,
    "remove_edges": _classify_remove_edges,
}


def extract(ops: list[Op]) -> list[Motif]:
    """
    Walk the Op history in reverse and return Motifs in
    reconstruction (generation) order.

    Each returned Motif carries a ``step`` index starting from 0.
    """
    motifs: list[Motif] = []
    for i, op in enumerate(reversed(ops)):
        classify = _CLASSIFIERS.get(op.kind)
        if classify is None:
            continue
        m = classify(op)
        motifs.append(Motif(
            kind=m.kind,
            node=m.node,
            preds=m.preds,
            succs=m.succs,
            meta=m.meta,
            step=i,
        ))
    return motifs
