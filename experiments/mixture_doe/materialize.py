"""mixture の提案を spec・コマンド・runner Plan に実体化する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Any

# ファイルを直接実行する CLI でも repository の training を参照する。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.runner.types import (
    Block, Plan, RescoreJob, TrainJob, plan_from_json, plan_to_json, validate_plan,
)

TARGETS = (("lay", "layered"), ("str", "structured"), ("spa", "spaghetti"))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {where} keys")


def _integer(value: Any, where: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")


def _text(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{where} must be a nonempty string")


def _validate(payload: Any) -> None:
    _keys(payload, {"version", "campaign_id", "campaign_revision", "context", "proposals"}, "envelope")
    _integer(payload["version"], "version", 1)
    if payload["version"] != 1:
        raise ValueError("unsupported envelope version")
    _text(payload["campaign_id"], "campaign_id")
    _integer(payload["campaign_revision"], "campaign_revision", 1)
    context = payload["context"]
    required = {"size", "prefix", "data_root", "max_offset", "baseline_dir", "response_scale"}
    if not isinstance(context, dict) or not required <= context.keys():
        raise ValueError("a non-null mixture context with size, prefix, data_root, max_offset, baseline_dir and response_scale is required")
    _integer(context["size"], "size")
    _integer(context["max_offset"], "max_offset")
    if context["size"] not in (24, 48):
        raise ValueError("unsupported size")
    if context["max_offset"] != {24: 20, 48: 37}[context["size"]]:
        raise ValueError("max_offset must match the fixed window for size")
    for field in ("prefix", "data_root", "baseline_dir"):
        _text(context[field], field)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", context["prefix"]):
        raise ValueError("prefix must be a run-name component")
    if context["response_scale"] != "excess_target":
        raise ValueError("mixture materialization requires excess_target responses")
    if "dataset_version" in context:
        _text(context["dataset_version"], "dataset_version")
    if "test_manifests" in context:
        _keys(context["test_manifests"], {t for t, _ in TARGETS}, "test_manifests")
        for digest in context["test_manifests"].values():
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("test_manifests must contain SHA256 digests")
    rows = payload["proposals"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("proposals must be a nonempty array")
    ids: set[str] = set()
    reserved: set[tuple[int, int]] = set()
    points: dict[int, tuple[int, dict[str, float]]] = {}
    candidates: dict[int, tuple[int, dict[str, float]]] = {}
    for row in rows:
        _keys(row, {"proposal_id", "candidate_id", "point", "factors", "seeds", "acquisition",
                    "observation_revision", "campaign_revision"}, "proposal")
        _text(row["proposal_id"], "proposal_id")
        if row["proposal_id"] in ids:
            raise ValueError("duplicate proposal_id")
        ids.add(row["proposal_id"])
        for field in ("point", "candidate_id", "observation_revision", "campaign_revision"):
            _integer(row[field], field)
        if (row["campaign_revision"] != payload["campaign_revision"]
                or row["observation_revision"] >= row["campaign_revision"]
                or row["acquisition"] not in ("ts", "ei", "ucb", "random", "fixed")):
            raise ValueError("invalid proposal metadata")
        factors = row["factors"]
        _keys(factors, {t for t, _ in TARGETS}, "mixture factors")
        weights = list(factors.values())
        if any(isinstance(w, bool) or not isinstance(w, (int, float))
               or not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("mixture weights must be finite and nonnegative")
        if abs(sum(weights) - 1) > 1e-9:
            raise ValueError("mixture weights must sum to one")
        point, candidate = row["point"], row["candidate_id"]
        if point in points and points[point] != (candidate, factors):
            raise ValueError("conflicting proposals for point")
        if candidate in candidates and candidates[candidate] != (point, factors):
            raise ValueError("conflicting proposals for candidate")
        points[point] = candidate, factors
        candidates[candidate] = point, factors
        if not isinstance(row["seeds"], list) or not row["seeds"]:
            raise ValueError("seeds must be a nonempty array")
        for seed in row["seeds"]:
            _integer(seed, "seed")
            if (point, seed) in reserved:
                raise ValueError("duplicate seed reservation")
            reserved.add((point, seed))
    _json(payload)


def _spec(factors: dict[str, float], size: int) -> dict[str, Any]:
    loops, gotos = {24: (3, 2), 48: (7, 4)}[size]
    components = []
    for target, family in TARGETS:
        if factors[target] == 0:
            continue
        params: dict[str, Any] = {"loop_count": loops, "goto_count": gotos}
        if family == "layered":
            params.update(max_layer_width=3, edge_prob=0.18, span_mode="uniform")
        elif family == "spaghetti":
            params["spaghetti_rate"] = 0.1
        components.append({"family": family, "weight": factors[target], "params": params})
    if len(components) == 1:
        return {"family": components[0]["family"], "num_nodes": size, "params": components[0]["params"]}
    return {"family": "mixture", "num_nodes": size, "params": {"components": components}}


def _command(*args: object) -> str:
    return shlex.join([str(arg) for arg in args])


def _fresh(path: Path) -> str:
    # 未検証の既存 dataset / bundle は上書きも暗黙の再利用もしない。
    return (f"if [ -e {shlex.quote(str(path))} ]; then "
            f"{_command('printf', '%s\\n', f'existing dataset/bundle requires verification: {path}')} >&2; exit 2; fi")


def materialize(payload: Any, out_dir: Path) -> dict[str, Any]:
    """全出力を検証してから新規ファイルのみを書く。同一出力は変更しない。"""
    _validate(payload)
    context = payload["context"]
    size, prefix = context["size"], context["prefix"]
    data = Path(context["data_root"])
    tests = {target: data / f"s{size}_{family}" for target, family in TARGETS}
    header = ["#!/usr/bin/env bash", "set -euo pipefail", "# repository root から実行する。"]
    generate, tokenize, collect = (header.copy() for _ in range(3))
    if "dataset_version" in context:
        # docs commits move HEAD without touching the generator, so compare the generator code
        # (cfg_reducer tree + prepare_tokens blob) at dataset_version with the same objects at HEAD.
        generate.append(_command("uv", "run", "python", "-c",
            "import subprocess,sys\n"
            "def obj(rev, path): return subprocess.check_output(['git','rev-parse',rev+':'+path],text=True).strip()\n"
            "paths=('cfg_reducer','training/prepare_tokens.py')\n"
            "sys.exit(0 if all(obj(sys.argv[1],p)==obj('HEAD',p) for p in paths) else 'generator code differs from dataset_version')",
            context["dataset_version"]))
    for target, digest in sorted(context.get("test_manifests", {}).items()):
        check = _command("uv", "run", "python", "-c",
            "import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); "
            "sys.exit(0 if actual == sys.argv[2] else 'fixed test manifest hash mismatch')",
            tests[target] / "manifest.json", digest)
        generate.append(check)
        tokenize.append(check)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in payload["proposals"]:
        grouped.setdefault(row["point"], []).append(row)
    files: dict[str, str] = {}
    blocks = []
    for point, rows in sorted(grouped.items()):
        spec_path = f"specs/spec_p{point}.json"
        files[spec_path] = _json(_spec(rows[0]["factors"], size))
        dataset = data / f"d{size}_p{point}"
        bundles = {target: data / f"tok_d{size}_p{point}_{target}" for target, _ in TARGETS}
        generate.append(_fresh(dataset))
        args: list[object] = ["uv", "run", "python", "-m", "cfg_reducer.dataset_v2",
                              "--spec", out_dir / spec_path, "--out", dataset]
        for split in ("train=800000:802150", "val=802150:802370", "test=802370:802400"):
            args.extend(("--split", split))
        for target, _ in TARGETS:
            args.extend(("--exclude-dataset", tests[target]))
        generate.append(_command(*args, "--allow-version-mismatch", "--allow-incomplete"))
        for target, _ in TARGETS:
            tokenize.append(_fresh(bundles[target]))
            tokenize.append(_command("uv", "run", "python", "-m", "training.prepare_tokens",
                "--dataset", dataset, "--test-dataset", tests[target], "--max-offset", context["max_offset"],
                "--out", bundles[target]))
        seeds = sorted(seed for row in rows for seed in row["seeds"])
        train, rescore = [], []
        for seed in seeds:
            name = f"{prefix}_s{size}_p{point}_mask_n{size}_s{seed}"
            train.append(TrainJob(name, 300, 20, 400, 400, seed, 1000 + seed, ("--ref-legal-mask",)))
            for target in ("str", "spa"):
                rescore.append(RescoreJob(name, str(bundles[target]),
                    f"{prefix}_s{size}_p{point}2{target}_mask_n{size}_s{seed}"))
        blocks.append(Block(str(bundles["lay"]), tuple(train), tuple(rescore)))
        collect.append(_command("uv", "run", "python", "-m", "training.mixture_doe", "collect",
            "--prefix", prefix, "--size", size, "--runs", "runs", "--data", data,
            "--points", point, "--seeds", ",".join(map(str, seeds)),
            "--baseline-dir", context["baseline_dir"], "--baseline-mode", "target",
            "--out", out_dir / f"obs_p{point}.json"))
    plan = Plan(f"{payload['campaign_id']}_r{payload['campaign_revision']}", tuple(blocks))
    validate_plan(plan)
    files["plan.json"] = plan_to_json(plan)
    restored = plan_from_json(files["plan.json"])
    validate_plan(restored)
    if restored != plan:
        raise ValueError("Plan round-trip mismatch")
    for name, lines in (("generate.sh", generate), ("tokenize.sh", tokenize), ("collect.sh", collect)):
        files[name] = "\n".join(lines) + "\n"
    manifest = {"version": 1, "campaign_id": payload["campaign_id"],
                "campaign_revision": payload["campaign_revision"], "context": context,
                "proposal_ids": [row["proposal_id"] for row in payload["proposals"]],
                "files": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                          for name, text in sorted(files.items())}}
    files["manifest.json"] = _json(manifest)
    # 衝突の事前検査は manifest を含む全ファイルに対して行う。
    for name, text in files.items():
        path = out_dir / name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != text.encode("utf-8"))):
            raise ValueError(f"conflicting existing file: {path}")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"conflicting existing directory: {parent}")
    for name, text in files.items():
        path = out_dir / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        materialize(json.loads(args.proposal.read_text(encoding="utf-8")), args.out_dir)
    except (ValueError, OSError) as exc:
        print(f"materialize: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
