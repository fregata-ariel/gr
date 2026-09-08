"""Spec-driven dataset CLI and provenance-based cross-dataset references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import store
from .buckets import (
    AcceptDecision, AcceptHook, AcceptanceState, BucketDimension, BucketPlan,
    Candidate, CFGReference, bucket_acceptor, measurement_acceptor, ranges_acceptor,
)
from .dataset import DEFAULT_GENERATOR, _git_state, _parse_split, build_dataset, cfg_edges
from .engine import GraphEngine
from .family_registry import load_plugins
from .generate_v2 import descriptor_for, normalize_spec, spec_from_json, spec_to_json


def load_references(paths: tuple[Path, ...], *, version: str) -> tuple[CFGReference, ...]:
    """Replay accepted manifest entries; IDs validate provenance, not CFG content."""
    references = []
    for path in paths:
        raw = (path / "manifest.json").read_bytes()
        dataset_id = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        generator = manifest["generator"]
        if generator["version"] != version:
            raise ValueError("generator version mismatch")
        config = generator["config"]
        if generator["name"] == DEFAULT_GENERATOR.name:
            descriptor = DEFAULT_GENERATOR
        elif generator["name"].startswith("cfg_v2:"):
            spec = spec_from_json(config["spec"])
            descriptor = descriptor_for(spec)
            config = {"spec": spec_to_json(normalize_spec(spec))}
        else:
            raise ValueError(f"unknown generator: {generator['name']!r}")
        if descriptor.name != generator["name"]:
            raise ValueError("generator name and spec family mismatch")
        for split in manifest["splits"].values():
            for sample in split["samples"]:
                engine = GraphEngine()
                nodes = descriptor.fn(engine, seed=sample["seed"], **config)
                provenance = {"source": "synthetic", "generator": {
                    "name": descriptor.name, "version": version,
                    "seed": sample["seed"], "config": config,
                }}
                if store.sample_id_for(provenance) != sample["sample_id"]:
                    raise ValueError("generator version mismatch: sample_id differs on replay")
                references.append(CFGReference(dataset_id, sample["sample_id"],
                                               tuple(nodes), tuple(cfg_edges(engine))))
    return tuple(references)


def _acceptor(splits: dict[str, tuple[int, int]], values: dict) -> AcceptHook:
    unknown = values.keys() - splits.keys()
    if unknown:
        raise ValueError(f"plans for unknown splits: {sorted(unknown)}")
    plans: dict[str, BucketPlan | None] = {}
    hooks: dict[str, AcceptHook] = {}
    split_ranges = {}
    for split in splits:
        value = values.get(split)
        plan = None
        hook = measurement_acceptor
        if value is not None:
            if not isinstance(value, dict) or value.keys() - {
                "dims", "target_per_bucket", "active", "ranges",
            }:
                raise ValueError("invalid plan fields")
            if any(key in value for key in ("dims", "target_per_bucket", "active")):
                plan = BucketPlan(
                    tuple(BucketDimension(d["feature"], tuple(d["cuts"]), tuple(d["labels"]))
                          for d in value["dims"]),
                    value["target_per_bucket"],
                    None if value.get("active") is None else tuple(tuple(row) for row in value["active"]),
                )
                hook = bucket_acceptor(plan)
            if "ranges" in value:
                hook = ranges_acceptor(hook, value["ranges"])
                split_ranges[split] = value["ranges"]
        plans[split], hooks[split] = plan, hook

    def accept(candidate: Candidate, state: AcceptanceState) -> AcceptDecision:
        return hooks[candidate.split](candidate, state)

    setattr(accept, "plans", plans)
    setattr(accept, "split_ranges", split_ranges)
    return accept


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=_parse_split, action="append", required=True)
    parser.add_argument("--version")
    parser.add_argument("--plugin", action="append", default=[])
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    load_plugins(tuple(args.plugin))
    spec = spec_from_json(json.loads(args.spec.read_text(encoding="utf-8")))
    splits = dict(args.split)
    if len(splits) != len(args.split):
        raise ValueError("duplicate split name")
    values = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else {}
    accept = _acceptor(splits, values)
    code = _git_state()
    version = args.version if args.version is not None else code["commit"]
    paths = tuple(args.exclude_dataset)
    exclude = load_references(paths, version=version)
    setattr(accept, "excluded_datasets", tuple(
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths
    ))
    manifest = build_dataset(
        args.out, splits, {"spec": spec_to_json(normalize_spec(spec))}, version,
        generator=descriptor_for(spec), accept=accept, exclude=exclude, code=code,
    )
    if not args.allow_incomplete and any(not s["complete"] for s in manifest["splits"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
