"""Spec-driven dataset CLI and provenance-based cross-dataset references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from random import Random
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
from .generator_types import GeneratorSpec, _copy_params


@dataclass(frozen=True)
class NodeCountDistribution:
    choices: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", tuple(self.choices))
        object.__setattr__(self, "weights", tuple(self.weights))
        if (not self.choices or any(type(n) is not int or n < 1 for n in self.choices)
                or tuple(sorted(set(self.choices))) != self.choices):
            raise ValueError("choices must be ascending distinct positive integers")
        if (len(self.weights) != len(self.choices)
                or any(isinstance(w, bool) or not isinstance(w, (int, float))
                       or not math.isfinite(w) or w <= 0 for w in self.weights)):
            raise ValueError("weights must be finite positive numbers matching choices")


@dataclass(frozen=True)
class DatasetSpec:
    family: str
    num_nodes: NodeCountDistribution
    specs: tuple[GeneratorSpec, ...]
    params: dict | None


def dataset_spec_from_json(value: dict) -> GeneratorSpec | DatasetSpec:
    """Keep integer specs on the original codec; eagerly validate every choice."""
    if not isinstance(value, dict) or not isinstance(value.get("num_nodes"), dict):
        return spec_from_json(value)
    if set(value) not in ({"family", "num_nodes", "params"},
                          {"family", "num_nodes", "params_by_num_nodes"}):
        raise ValueError("invalid distribution spec fields")
    dist = value["num_nodes"]
    if (set(dist) != {"choices", "weights"}
            or not isinstance(dist["choices"], list) or not isinstance(dist["weights"], list)):
        raise ValueError("distribution requires choices and weights arrays")
    counts = NodeCountDistribution(tuple(dist["choices"]), tuple(dist["weights"]))
    per_n = "params_by_num_nodes" in value
    if per_n and (not isinstance(value["params_by_num_nodes"], dict)
                  or set(value["params_by_num_nodes"]) != {str(n) for n in counts.choices}):
        raise ValueError("params_by_num_nodes keys must exactly match choices")
    specs = tuple(spec_from_json({"family": value["family"], "num_nodes": n,
                                 "params": value["params_by_num_nodes"][str(n)] if per_n else value["params"]})
                  for n in counts.choices)
    return DatasetSpec(specs[0].family, counts, specs, None if per_n else _copy_params(value["params"]))


def dataset_spec_to_json(spec: GeneratorSpec | DatasetSpec) -> dict:
    if isinstance(spec, GeneratorSpec):
        return spec_to_json(spec)
    result = {"family": spec.family, "num_nodes": {
        "choices": list(spec.num_nodes.choices), "weights": list(spec.num_nodes.weights)}}
    if spec.params is not None:
        result["params"] = _copy_params(spec.params)
    else:
        result["params_by_num_nodes"] = {str(s.num_nodes): spec_to_json(s)["params"] for s in spec.specs}
    return result


def resolve_sample_config(config: dict, seed: int) -> dict:
    if not isinstance(config["spec"]["num_nodes"], dict):
        return config
    spec = dataset_spec_from_json(config["spec"])
    assert isinstance(spec, DatasetSpec)
    maximum = max(spec.num_nodes.weights)
    scaled = tuple(w / maximum for w in spec.num_nodes.weights)
    draw = Random("gr:num_nodes:v1:" + str(seed)).random() * math.fsum(scaled)
    cumulative = 0.0
    selected = spec.specs[-1]
    for concrete, weight in zip(spec.specs, scaled):
        cumulative += weight
        if draw < cumulative:
            selected = concrete
            break
    return {"spec": spec_to_json(normalize_spec(selected))}


def load_references(paths: tuple[Path, ...], *, version: str,
                    allow_version_mismatch: bool = False) -> tuple[CFGReference, ...]:
    """Replay accepted manifest entries; IDs validate provenance, not CFG content.

    The excluded dataset must carry the same generator version unless
    allow_version_mismatch is set — only do that when the families it
    used are unchanged between the two versions (AGENTS.md version rule),
    because the CFGs are regenerated with the current code.
    """
    references = []
    for path in paths:
        raw = (path / "manifest.json").read_bytes()
        dataset_id = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        generator = manifest["generator"]
        if generator["version"] != version and not allow_version_mismatch:
            raise ValueError("generator version mismatch")
        config = generator["config"]
        if generator["name"] == DEFAULT_GENERATOR.name:
            descriptor = DEFAULT_GENERATOR
        elif generator["name"].startswith("cfg_v2:"):
            spec = dataset_spec_from_json(config["spec"])
            descriptor = descriptor_for(spec.specs[0] if isinstance(spec, DatasetSpec) else spec)
            config = {"spec": dataset_spec_to_json(spec)}
        else:
            raise ValueError(f"unknown generator: {generator['name']!r}")
        if descriptor.name != generator["name"]:
            raise ValueError("generator name and spec family mismatch")
        for split_name, split in manifest["splits"].items():
            for sample in split["samples"]:
                resolved = resolve_sample_config(config, sample["seed"]) if "spec" in config else config
                node_mode = "per_num_nodes" in split or isinstance(config.get("spec", {}).get("num_nodes"), dict)
                if node_mode and sample.get("requested") != resolved:
                    raise ValueError("requested config differs on replay")
                engine = GraphEngine()
                nodes = descriptor.fn(engine, seed=sample["seed"], **resolved)
                # replay under the manifest's own version: the check is
                # about the manifest's internal consistency, not ours
                provenance = {"source": "synthetic", "generator": {
                    "name": descriptor.name, "version": generator["version"],
                    "seed": sample["seed"], "config": resolved,
                }}
                if store.sample_id_for(provenance) != sample["sample_id"]:
                    raise ValueError("generator version mismatch: sample_id differs on replay")
                if node_mode:
                    payload = json.loads((path / split_name / f"{sample['sample_id']}.json").read_bytes())
                    if payload.get("provenance") != provenance:
                        raise ValueError("sample provenance differs on replay")
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
    parser.add_argument("--target-count", action="append", default=[], metavar="NAME=N")
    parser.add_argument("--plugin", action="append", default=[])
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[])
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help="accept excluded datasets built under another generator "
                             "version (only when their families are unchanged)")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    load_plugins(tuple(args.plugin))
    spec = dataset_spec_from_json(json.loads(args.spec.read_text(encoding="utf-8")))
    splits = dict(args.split)
    if len(splits) != len(args.split):
        raise ValueError("duplicate split name")
    targets = {}
    for text in args.target_count:
        name, sep, count = text.partition("=")
        if not sep or name not in splits or name in targets or not count.isdecimal() or int(count) < 1:
            raise ValueError("target-count requires a unique known split and positive integer")
        targets[name] = int(count)
    node_mode = isinstance(spec, DatasetSpec) or bool(targets)
    if node_mode and args.plan:
        raise ValueError("num_nodes bucket mode cannot be combined with --plan")
    values = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else {}
    accept = _acceptor(splits, values)
    code = _git_state()
    version = args.version if args.version is not None else code["commit"]
    paths = tuple(args.exclude_dataset)
    exclude = load_references(paths, version=version,
                              allow_version_mismatch=args.allow_version_mismatch)
    setattr(accept, "excluded_datasets", tuple(
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths
    ))
    manifest = build_dataset(
        args.out, splits, {"spec": dataset_spec_to_json(spec)}, version,
        generator=descriptor_for(spec.specs[0] if isinstance(spec, DatasetSpec) else spec),
        accept=accept, exclude=exclude, code=code,
        resolve_config=resolve_sample_config if isinstance(spec, DatasetSpec) else None,
        target_counts=targets or None, record_node_counts=node_mode,
    )
    if not args.allow_incomplete and any(not s["complete"] for s in manifest["splits"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
