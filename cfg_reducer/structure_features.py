"""Structural features of canonical MetaGraphs, without training dependencies."""

from __future__ import annotations

from statistics import mean

from .types import MetaGraph


def _walk_levels(mg: MetaGraph, depth: int = 0):
    """Yield (depth, level MetaGraph) for every level, top first."""
    yield depth, mg
    for sub in mg.subgraphs.values():
        yield from _walk_levels(sub, depth + 1)


def features_for(mg: MetaGraph) -> dict:
    levels = list(_walk_levels(mg))
    motifs = [m for _, g in levels for m in g.motifs]
    loops = [m for m in motifs if m.kind == "loop"]

    widths = [len(g.motifs) for _, g in levels]
    in_deg: list[int] = []
    out_deg: list[int] = []
    offsets: list[int] = []
    for _, g in levels:
        order = sorted(g.motifs, key=lambda m: m.step)
        pos = {m.step: i for i, m in enumerate(order)}
        incoming = {m.step: 0 for m in order}
        outgoing = {m.step: 0 for m in order}
        for src, dst in g.edges:
            incoming[dst] += 1
            outgoing[src] += 1
            offsets.append(pos[dst] - pos[src])
        in_deg.extend(incoming.values())
        out_deg.extend(outgoing.values())

    kinds = {k: sum(1 for m in motifs if m.kind == k)
             for k in ("entry", "linear", "merge", "loop")}
    scc_sizes = [len(m.meta["scc"]) for m in loops]

    return {
        "n_motifs": len(motifs),
        "n_entry": kinds["entry"],
        "n_linear": kinds["linear"],
        "n_merge": kinds["merge"],
        "n_loops": kinds["loop"],
        "merge_ratio": kinds["merge"] / len(motifs) if motifs else 0.0,
        "n_levels": len(levels),
        "max_depth": max(d for d, _ in levels),
        "n_nested_loops": sum(1 for d, g in levels if d >= 1
                              for m in g.motifs if m.kind == "loop"),
        "top_width": len(mg.motifs),
        "max_width": max(widths),
        "mean_width": mean(widths),
        "n_edges": len(offsets),
        "max_in_degree": max(in_deg) if in_deg else 0,
        "mean_in_degree": mean(in_deg) if in_deg else 0.0,
        "max_out_degree": max(out_deg) if out_deg else 0,
        "mean_out_degree": mean(out_deg) if out_deg else 0.0,
        "max_offset": max(offsets) if offsets else 0,
        "mean_offset": mean(offsets) if offsets else 0.0,
        "refs_per_motif": len(offsets) / len(motifs) if motifs else 0.0,
        "max_scc_size": max(scc_sizes) if scc_sizes else 0,
        "mean_scc_size": mean(scc_sizes) if scc_sizes else 0.0,
        "n_back_edges": sum(len(m.meta["back_edges"]) for m in loops),
    }


