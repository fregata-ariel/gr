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
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
from typing import Callable

import networkx as nx

from . import metagraph, motif, store
from .algorithm import ReductionAlgorithm
from .engine import GraphEngine
from .generate import GENERATOR_NAME, generate_cfg
from .types import MetaGraph
from .buckets import (AcceptHook, AcceptDecision, AcceptanceState, BucketPlan,
                      Candidate, CFGReference)
from .generator_types import GenerationRejected

# generator(engine, *, seed=..., **config) -> node ids
Generator = Callable[..., list[str]]


@dataclass(frozen=True)
class GeneratorDescriptor:
    """
    Identity of a CFG generator as recorded in provenance and manifest.
    Different generator families must never share a name, or their
    samples become indistinguishable in provenance (external review
    2026-09-04, 1-5).

    name  stable id written to provenance.generator.name
    fn    generator(engine, *, seed=..., **config) -> node ids
    """
    name: str
    fn: Generator


DEFAULT_GENERATOR = GeneratorDescriptor(GENERATOR_NAME, generate_cfg)


def _as_descriptor(generator: Generator | GeneratorDescriptor
                   ) -> GeneratorDescriptor:
    if isinstance(generator, GeneratorDescriptor):
        return generator
    return GeneratorDescriptor(getattr(generator, "__name__", "generator"),
                               generator)


# ── pipeline helpers ─────────────────────────

def cfg_edges(engine: GraphEngine) -> list[tuple[str, str]]:
    """Snapshot the current adjacency as a sorted edge list."""
    return sorted(
        (src, dst)
        for src in engine.node_ids()
        for dst in engine.successors(src)
    )


def cfg_nodes(engine: GraphEngine) -> list[str]:
    """Snapshot the node set; isolated nodes are part of the identity."""
    return sorted(engine.node_ids())


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

def _digraph(edges: list[tuple[str, str]],
             nodes: list[str] | None = None) -> nx.DiGraph:
    """Build the identity graph. Pass nodes so that isolated nodes count;
    edges alone silently drop them (external review 2026-09-04, 1-6)."""
    graph = nx.DiGraph()
    if nodes is not None:
        graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def fingerprint(edges: list[tuple[str, str]],
                nodes: list[str] | None = None) -> str:
    """WL hash for candidate bucketing — never proof of isomorphism."""
    with warnings.catch_warnings():
        # nx >= 3.5 emits hash-change notices for attribute-less graphs
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(_digraph(edges, nodes),
                                               iterations=3)


def is_structural_duplicate(
    edges_a: list[tuple[str, str]], edges_b: list[tuple[str, str]],
    nodes_a: list[str] | None = None, nodes_b: list[str] | None = None,
) -> bool:
    """Confirm directed isomorphism between two candidate CFGs."""
    return nx.is_isomorphic(_digraph(edges_a, nodes_a),
                            _digraph(edges_b, nodes_b))


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
    generator: Generator | GeneratorDescriptor = DEFAULT_GENERATOR,
    code: dict | None = None,
    *,
    accept: AcceptHook | None = None,
    exclude: tuple[CFGReference, ...] = (),
) -> dict:
    """
    Generate one sample per seed, drop structural duplicates across the
    whole dataset (first occurrence wins, in split insertion order),
    and write out/<split>/<sample_id>.json plus out/manifest.json.

    config must be exactly the generator's keyword arguments (it is
    recorded verbatim in each sample's provenance). A bare callable is
    wrapped into a GeneratorDescriptor named after the function; code
    (e.g. {"commit", "dirty"} from the CLI) is recorded in the manifest
    only, so sample_ids stay a function of the generator identity.

    accept and exclude independently enable selection accounting. Quota hooks
    expose .plan; a split-dispatching callable can expose .plans mapping split
    names to BucketPlan objects so even unvisited buckets are reported.
    """
    _validate_splits(splits)
    if accept is not None or exclude:
        return _build_selected(out_dir, splits, config, version, generator, code, accept, exclude)
    desc = _as_descriptor(generator)
    out = Path(out_dir)

    # fingerprint -> [(nodes, edges, sample_id)] across every split
    seen: dict[str, list[tuple[list[str], list[tuple[str, str]], str]]] = {}
    manifest_splits: dict[str, dict] = {}

    for split_name, (start, stop) in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        kept: list[dict] = []
        dropped: list[dict] = []

        for seed in range(start, stop):
            engine = GraphEngine()
            desc.fn(engine, seed=seed, **config)
            nodes, edges = cfg_nodes(engine), cfg_edges(engine)

            fp = fingerprint(edges, nodes)
            duplicate_of = next(
                (sid for prev_nodes, prev_edges, sid in seen.get(fp, [])
                 if is_structural_duplicate(edges, prev_edges,
                                            nodes, prev_nodes)),
                None,
            )
            if duplicate_of is not None:
                dropped.append({"seed": seed, "duplicate_of": duplicate_of})
                continue

            provenance = {
                "source": "synthetic",
                "generator": {
                    "name": desc.name,
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

            seen.setdefault(fp, []).append((nodes, edges, sample_id))
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
            "name": desc.name, "version": version, "config": config,
        },
        "splits": manifest_splits,
    }
    if code is not None:
        manifest["code"] = code
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


# ── CLI ──────────────────────────────────────

def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state() -> dict:
    """{"commit": short hash, "dirty": bool} of the working tree, so a
    manifest built from uncommitted generator code is recognisable."""
    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit or "unknown",
        "dirty": bool(status) if status is not None else True,
    }


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
    state = _git_state()
    manifest = build_dataset(
        args.out, splits, config, args.version or state["commit"],
        code=state,
    )
    for name, info in manifest["splits"].items():
        print(
            f"{name}: kept {info['kept']}, "
            f"dropped {info['dropped_duplicates']} duplicates"
        )




def _build_selected(out_dir, splits, config, version, generator, code,
                    accept: AcceptHook | None, exclude: tuple[CFGReference, ...]) -> dict:
    desc = _as_descriptor(generator)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    references: dict[str, list[CFGReference]] = {}
    seen: dict[str, list[CFGReference]] = {}
    for ref in sorted(exclude, key=lambda r: (r.dataset_id, r.sample_id, r.nodes, r.edges)):
        references.setdefault(fingerprint(list(ref.edges), list(ref.nodes)), []).append(ref)
    manifest_splits = {}
    plans = {}
    with (out / 'rejections.jsonl').open('w', encoding='utf-8') as rejected_file:
        for split, (start, stop) in splits.items():
            split_dir = out / split
            split_dir.mkdir(parents=True, exist_ok=True)
            # Dispatch closures may expose split-specific plans through .plans.
            plan: BucketPlan | None = getattr(accept, 'plans', {}).get(split, getattr(accept, 'plan', None))
            plans[split] = asdict(plan) if plan is not None else None
            ranges = getattr(accept, 'ranges', None)
            if ranges is not None:
                plans[split] = (plans[split] or {}) | {'ranges': dict(ranges)}
            counts: dict[str, int] = {}
            per_bucket = {bid: _bucket_stats(plan.target_per_bucket) for bid in plan.bucket_ids()} if plan else {}
            unbucketed = _bucket_stats(None)
            kept, dropped = [], []
            reasons: dict[str, int] = {}
            for seed in range(start, stop):
                engine = GraphEngine()
                bucket = realized = duplicate_of = duplicate_dataset = None
                reason = None
                try:
                    generated_nodes = desc.fn(engine, seed=seed, **config)
                except GenerationRejected as exc:
                    reason = exc.reason
                else:
                    # Generator contract places the entry first; preserve it for reducibility.
                    nodes, edges = tuple(generated_nodes), tuple(cfg_edges(engine))
                    if set(nodes) != set(cfg_nodes(engine)) or len(nodes) != len(set(nodes)):
                        raise ValueError('generator returned an invalid node snapshot')
                    fp = fingerprint(list(edges), list(nodes))
                    for pool, duplicate_reason in ((references, 'cross_dataset_duplicate'), (seen, 'duplicate')):
                        match = next((r for r in pool.get(fp, []) if is_structural_duplicate(
                            list(edges), list(r.edges), list(nodes), list(r.nodes))), None)
                        if match is not None:
                            duplicate_of = match.sample_id
                            duplicate_dataset = match.dataset_id if pool is references else None
                            reason = duplicate_reason
                            break
                    provenance = {'source': 'synthetic', 'generator': {
                        'name': desc.name, 'version': version, 'seed': seed, 'config': config}}
                    sample_id = store.sample_id_for(provenance)
                    requested = deepcopy(config)
                    if accept is not None or reason is None:
                        mg = reduce_to_metagraph(engine)
                        if accept is not None:
                            decision = accept(Candidate(split, seed, sample_id, requested, nodes, edges, mg),
                                              AcceptanceState(tuple(sorted(counts.items()))))
                            if not isinstance(decision, AcceptDecision) or type(decision.accepted) is not bool:
                                raise TypeError('accept must return AcceptDecision')
                            if (decision.accepted and decision.reason is not None) or (not decision.accepted and not decision.reason):
                                raise ValueError('inconsistent acceptance decision')
                            bucket, realized = decision.bucket, decision.realized
                            if reason is None and not decision.accepted:
                                reason = decision.reason
                        if reason is None:
                            store.save_sample(mg, provenance, split_dir / f'{sample_id}.json', sample_id)
                            seen.setdefault(fp, []).append(CFGReference('', sample_id, nodes, edges))
                            if bucket is not None:
                                counts[bucket] = counts.get(bucket, 0) + 1
                            kept.append({'seed': seed, 'sample_id': sample_id, 'requested': requested,
                                         'realized': realized, 'bucket': bucket})
                stats = unbucketed if bucket is None else per_bucket.setdefault(bucket, _bucket_stats(None))
                stats['attempts'] += 1
                stats['accepted' if reason is None else 'rejected'] += 1
                if reason is not None:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    if duplicate_of is not None:
                        dropped.append({'seed': seed, 'duplicate_of': duplicate_of})
                    row = {'seed': seed, 'split': split, 'reason': reason, 'bucket': bucket,
                           'realized': realized, 'duplicate_of': duplicate_of,
                           'duplicate_dataset': duplicate_dataset}
                    rejected_file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
            for stats in per_bucket.values():
                if stats['target'] is not None:
                    stats['missing'] = max(0, stats['target'] - stats['accepted'])
            attempts = stop - start
            manifest_splits[split] = {
                'seed_range': [start, stop], 'kept': len(kept), 'dropped_duplicates': len(dropped),
                'samples': kept, 'dropped': dropped, 'attempts': attempts, 'accepted': len(kept),
                'rejected': attempts - len(kept), 'acceptance_rate': len(kept) / attempts,
                'complete': all(s['missing'] in (None, 0) for s in per_bucket.values()),
                'per_bucket': per_bucket, 'unbucketed': unbucketed, 'rejected_by_reason': reasons,
            }
    manifest = {'schema_version': store.SCHEMA_VERSION,
                'generator': {'name': desc.name, 'version': version, 'config': config},
                'splits': manifest_splits,
                'selection': {'version': 1, 'plans': plans,
                              'excluded_datasets': sorted({r.dataset_id for r in exclude})}}
    for key in ('attempts', 'accepted', 'rejected'):
        manifest[key] = sum(s[key] for s in manifest_splits.values())
    if code is not None:
        manifest['code'] = code
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def _bucket_stats(target: int | None) -> dict:
    return {'target': target, 'attempts': 0, 'accepted': 0, 'rejected': 0, 'missing': target}


if __name__ == "__main__":
    main()
