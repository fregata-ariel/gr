from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfg_reducer import GraphEngine, ReductionAlgorithm, metagraph, motif
from cfg_reducer.gen import build_cfg
from cfg_reducer.linearize import encode
from cfg_reducer.types import MetaGraph


def main() -> int:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    discarded = 0
    deduped = 0
    written = 0
    seen: set[str] = set()

    with args.out.open("w", encoding="utf-8") as fh:
        for num_nodes in range(args.min_nodes, args.max_nodes + 1):
            for edge_prob in args.edge_probs:
                for seed in range(args.seeds_per_config):
                    generated += 1
                    record = _build_record(num_nodes, edge_prob, seed)
                    if record is None:
                        discarded += 1
                        continue

                    token_key = "\x1f".join(record["tokens"])
                    if token_key in seen:
                        deduped += 1
                        continue

                    seen.add(token_key)
                    fh.write(json.dumps(record, separators=(",", ":")))
                    fh.write("\n")
                    written += 1

    print(
        (
            f"generated={generated} discarded={discarded} "
            f"deduped={deduped} written={written}"
        ),
        file=sys.stderr,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-nodes", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--edge-probs", type=_parse_edge_probs, required=True)
    parser.add_argument("--seeds-per-config", type=int, required=True)
    return parser.parse_args()


def _parse_edge_probs(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _build_record(num_nodes: int, edge_prob: float, seed: int) -> dict | None:
    engine = build_cfg(num_nodes=num_nodes, edge_prob=edge_prob, seed=seed)
    edges = _graph_edges(engine)
    mg = _extract_metagraph(engine, "N00")
    tokens = encode(mg)

    if not _passes_canonicality_self_check(edges, tokens, seed):
        return None

    stats = _stats_for(mg, tokens)
    prob_text = format(edge_prob, "g")
    return {
        "id": f"n{num_nodes}_p{prob_text}_s{seed}",
        "gen": {
            "num_nodes": num_nodes,
            "edge_prob": edge_prob,
            "seed": seed,
        },
        "tokens": tokens,
        "stats": stats,
    }


def _extract_metagraph(engine, entry: str | None) -> MetaGraph:
    algorithm = ReductionAlgorithm(engine, entry=entry)
    while algorithm.step() is not None:
        pass
    motifs = motif.extract(engine.history)
    return metagraph.build(motifs)


def _graph_edges(engine: GraphEngine) -> list[tuple[str, str]]:
    return [
        (src, dst)
        for src in sorted(engine.node_ids())
        for dst in sorted(engine.successors(src))
    ]


def _passes_canonicality_self_check(
    edges: list[tuple[str, str]],
    tokens: list[str],
    seed: int,
) -> bool:
    node_ids = sorted({node for edge in edges for node in edge})
    shuffled = list(node_ids)
    random.Random(seed).shuffle(shuffled)
    mapping = dict(zip(node_ids, shuffled, strict=True))

    permuted = GraphEngine()
    for node_id in sorted(node_ids):
        permuted.add_node(node_id)

    for src, dst in edges:
        permuted.add_edge(mapping[src], mapping[dst])

    return encode(_extract_metagraph(permuted, mapping["N00"])) == tokens


def _stats_for(mg: MetaGraph, tokens: list[str]) -> dict[str, int | dict[str, int]]:
    counts = _count_motifs(mg)
    depth, max_width = _top_level_shape(mg)
    return {
        "n_motifs": sum(counts.values()),
        "kinds": dict(counts),
        "n_tokens": len(tokens),
        "depth": depth,
        "max_width": max_width,
        "loop_nest": _loop_nest(mg),
    }


def _count_motifs(mg: MetaGraph) -> Counter[str]:
    counts = Counter(motif_.kind for motif_ in mg.motifs)
    for subgraph in mg.subgraphs.values():
        counts.update(_count_motifs(subgraph))
    return counts


def _top_level_shape(mg: MetaGraph) -> tuple[int, int]:
    if not mg.motifs:
        return 0, 0

    parents: dict[int, set[int]] = {motif.step: set() for motif in mg.motifs}
    children: dict[int, list[int]] = {motif.step: [] for motif in mg.motifs}
    for src, dst in set(mg.edges):
        if src in children and dst in parents and src != dst:
            parents[dst].add(src)
            children[src].append(dst)

    indegree = {step: len(parent_set) for step, parent_set in parents.items()}
    current = [step for step, degree in indegree.items() if degree == 0]
    depth = 0
    max_width = 0

    while current:
        depth += 1
        max_width = max(max_width, len(current))
        next_layer: list[int] = []
        for step in current:
            for child in children[step]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_layer.append(child)
        current = next_layer

    return depth, max_width


def _loop_nest(mg: MetaGraph) -> int:
    best = 0
    for step, subgraph in mg.subgraphs.items():
        del step
        best = max(best, 1 + _loop_nest(subgraph))
    return best


if __name__ == "__main__":
    raise SystemExit(main())
