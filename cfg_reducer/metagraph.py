"""Convert hierarchical Motifs into dependency DAGs."""

from __future__ import annotations

from .types import Motif, MetaGraph


def build(motifs: list[Motif]) -> MetaGraph:
    node_to_step: dict[str, int] = {}
    for m in motifs:
        if m.kind == "loop":
            for n in m.meta["scc"]:
                node_to_step[n] = m.step
        elif m.node is not None:
            node_to_step[m.node] = m.step

    edge_set: set[tuple[int, int]] = set()
    for m in motifs:
        for pred in m.preds:
            if pred in node_to_step:
                src_step = node_to_step[pred]
                if src_step != m.step:
                    edge_set.add((src_step, m.step))

        for succ in m.succs:
            if succ in node_to_step:
                dst_step = node_to_step[succ]
                if dst_step != m.step:
                    edge_set.add((m.step, dst_step))

    subgraphs: dict[int, MetaGraph] = {}
    for m in motifs:
        if m.kind == "loop" and m.children:
            subgraphs[m.step] = build(list(m.children))

    return MetaGraph(
        motifs=tuple(motifs),
        edges=tuple(sorted(edge_set)),
        subgraphs=subgraphs,
    )
