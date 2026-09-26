#!/usr/bin/env python3
"""Summarize controlled CFG-eval runs into Markdown + JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.controlled_eval import summarize

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
CELL_BY_KEY: dict[str, dict[str, Any]] = {c["key"]: c for c in CELLS}

OOD_PAIRS: list[tuple[str, str, str]] = [
    ("layered_id", "layered_lay2str", "layered -> lay2str"),
    ("layered_id", "layered_lay2spa", "layered -> lay2spa"),
    ("structured_id", "structured_str2lay", "structured -> str2lay"),
    ("structured_id", "structured_str2spa", "structured -> str2spa"),
    ("depth_id", "depth_ood", "depth1 -> depth"),
    ("merge_id", "merge_ood", "merge2 -> merge"),
]


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


def cell_table(cell: dict[str, Any], summ: dict[str, Any] | None) -> list[str]:
    lines = [f"### {cell['label']}", ""]
    if summ is None:
        return lines + ["_pending_", ""]
    lines += [
        "| config | test NLL (mean ± sd) | paired ΔNLL vs base (sign-consistent?) | edge acc | WF (kind) |",
        "|---|---|---|---|---|",
    ]
    for cfg in cell["configs"]:
        c = _cfg(summ, cfg)
        lines.append(f"| {cfg} | {fmt_nll(c)} | {fmt_delta(c)} | {fmt_edge(c)} | {fmt_wf(c)} |")
    lines.append("")
    return lines


def ood_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "## OOD shift",
        "",
        "| source -> target | config | ID NLL | OOD NLL | OOD − ID | ID WF |",
        "|---|---|---|---|---|---|",
    ]
    for src, tgt, label in OOD_PAIRS:
        a, b = results.get(src), results.get(tgt)
        if a is None or b is None:
            lines.append(f"| {label} | — | _pending_ | _pending_ | — | — |")
            continue
        for cfg in CELL_BY_KEY[src]["configs"]:
            ca, cb = _cfg(a, cfg), _cfg(b, cfg)
            na = (ca.get("test_nll_per_token") or {}).get("mean")
            nb = (cb.get("test_nll_per_token") or {}).get("mean")
            id_nll = "—" if na is None else f"{na:.4f}"
            ood_nll = "—" if nb is None else f"{nb:.4f}"
            diff = "—" if na is None or nb is None else f"{nb - na:+.4f}"
            lines.append(f"| {label} | {cfg} | {id_nll} | {ood_nll} | {diff} | {fmt_wf(ca)} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--out", default="runs/c_summary.md")
    args = ap.parse_args()

    results: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    for cell in CELLS:
        try:
            res = summarize(
                args.runs, cell["prefix"], args.size, cell["tokens"],
                cell["configs"], BASELINE, SEEDS,
            )
            results[cell["key"]] = res
            raw[cell["key"]] = res
        except Exception as exc:  # noqa: BLE001 - report any missing/pending cell
            results[cell["key"]] = None
            raw[cell["key"]] = {"status": "pending", "error": str(exc)}

    lines = [
        "# Controlled-eval summary",
        "",
        f"runs={args.runs}, size={args.size}, seeds={SEEDS}, baseline={BASELINE}",
        "",
    ]
    for cell in CELLS:
        lines += cell_table(cell, results[cell["key"]])
    lines += ood_table(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
