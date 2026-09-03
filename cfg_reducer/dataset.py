"""
Seed-based synthetic dataset generation.

generate -> reduce -> extract -> build -> canonical sample JSON, with
split separation by disjoint seed ranges and structural dedup
(WL-hash bucketing + directed-isomorphism confirmation, handoff Q&A
A3-4).  Layout and manifest format: docs/design/dataset_generation.md.
"""

from __future__ import annotations
import argparse
import json
import subprocess
import warnings
from pathlib import Path
from typing import Callable

import networkx as nx

from . import metagraph, motif, store
from .algorithm import ReductionAlgorithm
from .engine import GraphEngine
from .generate import GENERATOR_NAME, generate_cfg
from .types import MetaGraph

# generator(engine, *, seed=..., **config) -> node ids
Generator = Callable[..., list[str]]


# ── pipeline helpers ─────────────────────────

def cfg_edges(engine: GraphEngine) -> list[tuple[str, str]]:
    """Snapshot the current adjacency as a sorted edge list."""
    return sorted(
        (src, dst)
        for src in engine.node_ids()
        for dst in engine.successors(src)
    )


def reduce_to_metagraph(engine: GraphEngine) -> MetaGraph:
    """Run the full pipeline on a built engine (reduces it to empty)."""
    algorithm = ReductionAlgorithm(engine)
    while algorithm.step() is not None:
        pass
    return metagraph.build(motif.extract(engine.history))


# ── structural dedup (handoff Q&A A3-4) ──────
#
# All nodes share one node_type today, so plain directed structure is
# the full identity.  When node types diversify, pass node attributes
# to both the hash (node_attr) and the matcher (node_match).

def _digraph(edges: list[tuple[str, str]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def fingerprint(edges: list[tuple[str, str]]) -> str:
    """WL hash for candidate bucketing — never proof of isomorphism."""
    with warnings.catch_warnings():
        # nx >= 3.5 emits hash-change notices for attribute-less graphs
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(_digraph(edges), iterations=3)


def is_structural_duplicate(
    edges_a: list[tuple[str, str]], edges_b: list[tuple[str, str]],
) -> bool:
    """Confirm directed isomorphism between two candidate CFGs."""
    return nx.is_isomorphic(_digraph(edges_a), _digraph(edges_b))


# ── dataset builder ──────────────────────────

def _validate_splits(splits: dict[str, tuple[int, int]]) -> None:
    for name, (start, stop) in splits.items():
        if start >= stop:
            raise ValueError(f"split {name!r}: empty seed range [{start}, {stop})")
    spans = sorted((rng, name) for name, rng in splits.items())
    for ((_, stop_a), name_a), ((start_b, _), name_b) in zip(spans, spans[1:]):
        if start_b < stop_a:
            raise ValueError(
                f"seed ranges overlap between splits {name_a!r} and {name_b!r}"
            )


def build_dataset(
    out_dir: str | Path,
    splits: dict[str, tuple[int, int]],
    config: dict,
    version: str,
    generator: Generator = generate_cfg,
) -> dict:
    """
    Generate one sample per seed, drop structural duplicates across the
    whole dataset (first occurrence wins, in split insertion order),
    and write out/<split>/<sample_id>.json plus out/manifest.json.

    config must be exactly the generator's keyword arguments (it is
    recorded verbatim in each sample's provenance).
    """
    _validate_splits(splits)
    out = Path(out_dir)

    # fingerprint -> [(edges, sample_id)] across every split
    seen: dict[str, list[tuple[list[tuple[str, str]], str]]] = {}
    manifest_splits: dict[str, dict] = {}

    for split_name, (start, stop) in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        kept: list[dict] = []
        dropped: list[dict] = []

        for seed in range(start, stop):
            engine = GraphEngine()
            generator(engine, seed=seed, **config)
            edges = cfg_edges(engine)

            fp = fingerprint(edges)
            duplicate_of = next(
                (sid for prev_edges, sid in seen.get(fp, [])
                 if is_structural_duplicate(edges, prev_edges)),
                None,
            )
            if duplicate_of is not None:
                dropped.append({"seed": seed, "duplicate_of": duplicate_of})
                continue

            provenance = {
                "source": "synthetic",
                "generator": {
                    "name": GENERATOR_NAME,
                    "version": version,
                    "seed": seed,
                    "config": config,
                },
            }
            sample_id = store.sample_id_for(provenance)
            mg = reduce_to_metagraph(engine)
            store.save_sample(
                mg, provenance, split_dir / f"{sample_id}.json", sample_id
            )

            seen.setdefault(fp, []).append((edges, sample_id))
            kept.append({"seed": seed, "sample_id": sample_id})

        manifest_splits[split_name] = {
            "seed_range": [start, stop],
            "kept": len(kept),
            "dropped_duplicates": len(dropped),
            "samples": kept,
            "dropped": dropped,
        }

    manifest = {
        "schema_version": store.SCHEMA_VERSION,
        "generator": {
            "name": GENERATOR_NAME, "version": version, "config": config,
        },
        "splits": manifest_splits,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


# ── CLI ──────────────────────────────────────

def _git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_split(text: str) -> tuple[str, tuple[int, int]]:
    name, sep, span = text.partition("=")
    start_s, sep2, stop_s = span.partition(":")
    if not sep or not sep2 or not name:
        raise argparse.ArgumentTypeError(
            f"expected NAME=START:STOP, got {text!r}"
        )
    return name, (int(start_s), int(stop_s))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cfg_reducer.dataset",
        description="Generate a synthetic MetaGraph dataset.",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--split", action="append", required=True, metavar="NAME=START:STOP",
        help="split name and seed range, e.g. train=0:800 (repeatable)",
    )
    parser.add_argument("--num-nodes", type=int, default=12)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument(
        "--version", default=None,
        help="generator version tag (default: current git commit)",
    )
    args = parser.parse_args(argv)

    splits = dict(_parse_split(s) for s in args.split)
    config = {"num_nodes": args.num_nodes, "edge_prob": args.edge_prob}
    manifest = build_dataset(
        args.out, splits, config, args.version or _git_version()
    )
    for name, info in manifest["splits"].items():
        print(
            f"{name}: kept {info['kept']}, "
            f"dropped {info['dropped_duplicates']} duplicates"
        )


if __name__ == "__main__":
    main()
