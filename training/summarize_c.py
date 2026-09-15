#!/usr/bin/env python3
"""Summarise the C (generator v2 / structural OOD) runs into Markdown + JSON.

    uv run python -m training.summarize_c --out runs/c_summary.md [--cells k1,k2] [--json-only]

Cell prefixes / token bundles follow docs/design/generator_v2.md §10-§11. Drafted
by OpenCode (deepseek-v4.1-flash) as a prototype and reviewed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.controlled_eval import run_dir_name, summarize

SEEDS = [0, 1, 2]
BASELINE = "base"

CELLS: list[dict[str, Any]] = [
    {"key": "layered_id", "label": "Layered (ID)", "prefix": "c_layered_",
     "tokens": "data/tok_c_layered", "configs": ["base", "mask", "ptr"]},
    {"key": "layered_lay2str", "label": "Layered -> structured (family OOD)", "prefix": "c_layered2lay2str_",
     "tokens": "data/tok_c_lay2str", "configs": ["base", "mask", "ptr"]},
    {"key": "layered_lay2spa", "label": "Layered -> spaghetti (family OOD)", "prefix": "c_layered2lay2spa_",
     "tokens": "data/tok_c_lay2spa", "configs": ["base", "mask", "ptr"]},
    {"key": "structured_id", "label": "Structured (ID)", "prefix": "c_structured_",
     "tokens": "data/tok_c_structured", "configs": ["base", "mask"]},
    {"key": "structured_str2lay", "label": "Structured -> layered (family OOD')", "prefix": "c_structured2str2lay_",
     "tokens": "data/tok_c_str2lay", "configs": ["base", "mask"]},
    {"key": "structured_str2spa", "label": "Structured -> spaghetti (family OOD')", "prefix": "c_structured2str2spa_",
     "tokens": "data/tok_c_str2spa", "configs": ["base", "mask"]},
    {"key": "depth_id", "label": "Depth-1 (ID)", "prefix": "c_depth1_",
     "tokens": "data/tok_c_depth_id", "configs": ["base", "mask"]},
    {"key": "depth_ood", "label": "Depth-1 -> depth-3 (depth OOD)", "prefix": "c_depth12depth_",
     "tokens": "data/tok_c_depth", "configs": ["base", "mask"]},
    {"key": "merge_id", "label": "Merge-2 (ID)", "prefix": "c_merge2_",
     "tokens": "data/tok_c_merge_id", "configs": ["base", "mask"]},
    {"key": "merge_ood", "label": "Merge-2 -> merge-4 (degree OOD)", "prefix": "c_merge22merge_",
     "tokens": "data/tok_c_merge", "configs": ["base", "mask"]},
    {"key": "balanced_id", "label": "Balanced-k", "prefix": "c_balanced_",
     "tokens": "data/tok_c_balanced", "configs": ["base", "mask"]},
]
CELLS += [
    {"key": "mixed_id", "label": "Mixed (ID)", "prefix": "c_mixed_",
     "tokens": "data/tok_c_mixed", "configs": ["base", "mask"]},
    *[{"key": f"mixed_mix2{target}", "label": f"Mixed -> {target} (OOD)",
       "prefix": f"c_mixed2mix2{target}_", "tokens": f"data/tok_c_mix2{target}",
       "configs": ["base", "mask"]} for target in ("lay", "str", "spa", "depth", "merge")],
    *[{"key": f"{source}_{short}2{target}", "label": f"{source} -> {target} (OOD)",
       "prefix": f"c_{source}2{short}2{target}_", "tokens": f"data/tok_c_{short}2{target}",
       "configs": ["base", "mask", "ptr"] if source == "layered" else ["base", "mask"]}
      for source, short in (("layered", "lay"), ("structured", "str"))
      for target in ("depth", "merge")],
]
CELL_BY_KEY: dict[str, dict[str, Any]] = {c["key"]: c for c in CELLS}

OOD_PAIRS: list[tuple[str, str, str]] = [
    ("layered_id", "layered_lay2str", "layered -> lay2str"),
    ("layered_id", "layered_lay2spa", "layered -> lay2spa"),
    ("structured_id", "structured_str2lay", "structured -> str2lay"),
    ("structured_id", "structured_str2spa", "structured -> str2spa"),
    ("depth_id", "depth_ood", "depth1 -> depth"),
    ("merge_id", "merge_ood", "merge2 -> merge"),
]

OOD_PAIRS += [
    ("mixed_id", f"mixed_mix2{target}", f"mixed -> {target}")
    for target in ("lay", "str", "spa", "depth", "merge")
] + [
    (f"{source}_id", f"{source}_{short}2{target}", f"{source} -> {target}")
    for source, short in (("layered", "lay"), ("structured", "str"))
    for target in ("depth", "merge")
]

TARGET_CELLS = {
    "layered": {"layered": "layered_id", "structured": "structured_str2lay", "mixed": "mixed_mix2lay"},
    "structured": {"layered": "layered_lay2str", "structured": "structured_id", "mixed": "mixed_mix2str"},
    "spaghetti": {"layered": "layered_lay2spa", "structured": "structured_str2spa", "mixed": "mixed_mix2spa"},
    "depth3": {"layered": "layered_lay2depth", "structured": "structured_str2depth",
               "depth1": "depth_ood", "mixed": "mixed_mix2depth"},
    "merge4": {"layered": "layered_lay2merge", "structured": "structured_str2merge",
               "merge2": "merge_ood", "mixed": "mixed_mix2merge"},
}


def _cfg(summ: dict[str, Any], cfg: str) -> dict[str, Any]:
    return summ.get("summary", {}).get(cfg, {})


def fmt_nll(c: dict[str, Any]) -> str:
    nll = c.get("test_nll_per_token") or {}
    if nll.get("mean") is None:
        return "—"
    return f"{nll['mean']:.4f} ± {nll.get('sd', 0.0):.4f}"


def fmt_delta(c: dict[str, Any]) -> str:
    d = c.get("paired_delta_vs_baseline")
    if not d or d.get("mean") is None:
        return "—"
    flag = "yes" if d.get("all_same_sign") else "no"
    return f"{d['mean']:+.4f} ({flag})"


def fmt_edge(c: dict[str, Any]) -> str:
    e = (c.get("edge_accuracy") or {}).get("mean")
    return "—" if e is None else f"{e:.3f}"


def fmt_wf(c: dict[str, Any]) -> str:
    kind = c.get("wf_kind", "?")
    m = (c.get("wf") or {}).get("mean")
    return f"— ({kind})" if m is None else f"{m * 100:.1f}% ({kind})"


def ood_gap(a: dict[str, Any], b: dict[str, Any]) -> str:
    d: list[float] = []
    for cfg in ("base", "mask"):
        na = (_cfg(a, cfg).get("test_nll_per_token") or {}).get("mean")
        nb = (_cfg(b, cfg).get("test_nll_per_token") or {}).get("mean")
        if na is None or nb is None:
            return "—"
        d.append(nb - na)
    return f"{d[1] - d[0]:+.4f}"


def parse_cells(value: str) -> list[str]:
    keys = [k.strip() for k in value.split(",") if k.strip()]
    unknown = [k for k in keys if k not in CELL_BY_KEY]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown cell key(s): {', '.join(unknown)}; valid keys: {', '.join(CELL_BY_KEY)}")
    return keys


def _available(summ: dict[str, Any]) -> bool:
    return summ.get("status", "complete") in ("complete", "partial")


def counts(summ: dict[str, Any], cfg: str) -> str:
    n_seeds = len(_cfg(summ, cfg).get("seeds", []))
    return f"{n_seeds} / {summ.get('n_test', 0) if n_seeds else 0}"


def coverage(tokens: Path) -> dict | None:
    meta = json.loads((tokens / "meta.json").read_text(encoding="utf-8"))
    stats = meta.get("counts", {}).get("test")
    if stats is None:
        return None
    exclusions = [r for r in meta.get("exclusions", []) if r["split"] == "test"]
    return {**{k: stats[k] for k in ("retained", "raw", "excluded")},
            **{reason: sum(r["reason"] == reason for r in exclusions)
               for reason in ("over_window", "over_length")}}


def collect_cell(cell: dict[str, Any], runs: Path, size: int,
                 seeds: list[int], *, strict: bool = True) -> dict[str, Any]:
    missing = []
    for cfg in cell["configs"]:
        for seed in seeds:
            directory = runs / run_dir_name(cell["prefix"], cfg, size, seed)
            if not directory.is_dir():
                missing.append(str(directory))
            elif not (directory / "test_scores.jsonl").is_file():
                missing.append(str(directory / "test_scores.jsonl"))
    result: dict[str, Any] = {"status": "pending", "incomplete": bool(missing), "missing": missing}
    try:
        result["coverage"] = coverage(Path(cell["tokens"]))
        if strict and missing:
            return result
        report = summarize(runs, cell["prefix"], size, cell["tokens"],
                           cell["configs"], BASELINE, seeds)
        result.update(report)
        result["status"] = "partial" if missing else "complete"
        for cfg, entry in report["summary"].items():
            paired = entry.get("paired_delta_vs_baseline", {}).get("per_seed", {})
            entry["n_seeds"] = len(entry["seeds"])
            entry["n_test"] = report["n_test"]
            entry["paired_n_samples"] = {seed: delta["n"] for seed, delta in paired.items()}
        return result
    except FileNotFoundError as exc:
        result.update(status="pending", incomplete=True, error=str(exc))
    except ValueError as exc:
        result.update(status="error", error=str(exc))
    return result


def cell_table(cell: dict[str, Any], summ: dict[str, Any]) -> list[str]:
    cov = summ.get("coverage")
    heading = f"### {cell['label']}"
    if not cell["key"].endswith("_id"):
        heading += (f" — retained {cov['retained']} / raw {cov['raw']}, excluded {cov['excluded']} "
                    f"(over_window {cov['over_window']}, over_length {cov['over_length']})"
                    if cov is not None else " — retained / raw / excluded: n/a")
    lines = [heading, ""]
    if summ.get("missing"):
        lines += ["incomplete: " + ", ".join(summ["missing"]), ""]
    if not _available(summ):
        return lines + [f"_{summ['status']}_: {summ.get('error', 'ファイル未到着')}", ""]
    lines += [
        "| config | seeds / test samples (per seed) | paired samples (seed:n) | test NLL (mean ± sd) | paired ΔNLL vs base (sign-consistent?) | edge acc | source unconditional WF (kind) |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in cell["configs"]:
        c = _cfg(summ, cfg)
        paired = c.get("paired_n_samples", {})
        pair_text = ", ".join(f"{seed}:{n}" for seed, n in paired.items()) or "—"
        lines.append(f"| {cfg} | {counts(summ, cfg)} | {pair_text} | {fmt_nll(c)} | {fmt_delta(c)} | {fmt_edge(c)} | {fmt_wf(c)} |")
    return lines + [""]


def ood_table(results: dict[str, Any], selected: set[str]) -> list[str]:
    lines = ["## OOD shift", "",
        "| source -> target | config | ID seeds / test samples | OOD seeds / test samples | ID NLL | OOD NLL | OOD − ID | OOD − ID (mask − base) | source unconditional ID WF |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for src, tgt, label in OOD_PAIRS:
        if src not in selected or tgt not in selected:
            continue
        a, b = results[src], results[tgt]
        if not _available(a) or not _available(b):
            continue
        gap = ood_gap(a, b)
        for cfg in CELL_BY_KEY[src]["configs"]:
            ca, cb = _cfg(a, cfg), _cfg(b, cfg)
            na = (ca.get("test_nll_per_token") or {}).get("mean")
            nb = (cb.get("test_nll_per_token") or {}).get("mean")
            id_nll = "—" if na is None else f"{na:.4f}"
            ood_nll = "—" if nb is None else f"{nb:.4f}"
            diff = "—" if na is None or nb is None else f"{nb - na:+.4f}"
            lines.append(f"| {label} | {cfg} | {counts(a, cfg)} | {counts(b, cfg)} | {id_nll} | {ood_nll} | {diff} | {gap} | {fmt_wf(ca)} |")
    return lines + [""]


def target_table(results: dict[str, Any]) -> list[str]:
    sources = ("layered", "structured", "depth1", "merge2", "mixed")
    lines = ["## target 別比較", "", "mask NLL (mean ± sd; seeds / test samples per seed)。同率最良はすべて太字。", "",
             "| target | " + " | ".join(sources) + " |", "|---|---|---|---|---|---|"]
    for target, mapping in TARGET_CELLS.items():
        available = {source: results[key] for source, key in mapping.items()
                     if key in results and _available(results[key])
                     and _cfg(results[key], "mask").get("test_nll_per_token", {}).get("mean") is not None}
        best = min((_cfg(r, "mask")["test_nll_per_token"]["mean"] for r in available.values()), default=None)
        cells = []
        for source in sources:
            if source not in available:
                cells.append("—")
                continue
            rep = available[source]
            cfg = _cfg(rep, "mask")
            value = f"{fmt_nll(cfg)} ({counts(rep, 'mask')})"
            cells.append(f"**{value}**" if cfg["test_nll_per_token"]["mean"] == best else value)
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    return lines + [""]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--out", default="runs/c_summary.md")
    ap.add_argument("--cells", type=parse_cells, default=None)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--strict", dest="strict", action="store_true", default=True)
    mode.add_argument("--allow-partial", dest="strict", action="store_false")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)
    seeds = list(dict.fromkeys(int(s) for s in args.seeds.split(",")))
    cells = [CELL_BY_KEY[k] for k in args.cells] if args.cells else CELLS
    results = {cell["key"]: collect_cell(cell, Path(args.runs), args.size, seeds, strict=args.strict)
               for cell in cells}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.json_only:
        return
    lines = ["# Controlled-eval summary", "",
             f"runs={args.runs}, size={args.size}, seeds={seeds}, baseline={BASELINE}, strict={args.strict}", ""]
    for cell in cells:
        lines += cell_table(cell, results[cell["key"]])
    lines += ood_table(results, set(results))
    lines += target_table(results)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
