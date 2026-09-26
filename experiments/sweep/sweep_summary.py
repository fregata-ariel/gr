#!/usr/bin/env python3
"""Milestone sweep summary: per-size cells -> Markdown + JSON.

Calls training.controlled_eval.summarize for each size and cell and
reports test NLL, OOD shift, WF and edge accuracy. Missing cells are
reported as pending and never crash the sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.controlled_eval import summarize

SIZES = (12, 16, 24, 32, 48)
CONFIG = "mask"
BASELINE = "mask"

CELLS = (
    ("lay_id", "lay_", "lay", "lay ID"),
    ("mix_id", "mix_", "mix", "mix ID"),
    ("lay_str", "lay2str_", "lay2str", "lay->str"),
    ("mix_str", "mix2str_", "mix2str", "mix->str"),
    ("lay_spa", "lay2spa_", "lay2spa", "lay->spa"),
    ("mix_spa", "mix2spa_", "mix2spa", "mix->spa"),
    ("mix_lay", "mix2lay_", "mix2lay", "mix->lay"),
)
NLL_KEYS = tuple(key for key, _, _, _ in CELLS)
LABEL = {key: lbl for key, _, _, lbl in CELLS}
SHIFTS = (
    ("lay_str", "lay_id", "lay->str - lay ID"),
    ("mix_str", "mix_id", "mix->str - mix ID"),
    ("lay_spa", "lay_id", "lay->spa - lay ID"),
    ("mix_spa", "mix_id", "mix->spa - mix ID"),
    ("mix_lay", "lay_id", "mix->lay - lay ID"),
)


def extract(rep: dict[str, Any]) -> dict[str, Any]:
    entry = (rep.get("summary") or {}).get(CONFIG) or {}
    nll = entry.get("test_nll_per_token") or {}
    wf = entry.get("wf") or {}
    return {
        "nll_mean": nll.get("mean"),
        "nll_sd": nll.get("sd"),
        "edge_mean": (entry.get("edge_accuracy") or {}).get("mean"),
        "wf_mean": wf.get("mean"),
        "wf_kind": entry.get("wf_kind", "unknown"),
        "n_test": rep.get("n_test"),
        "n_seeds": len(nll.get("per_seed") or []),
    }


def fmt_nll(entry: dict[str, Any] | None) -> str:
    if not entry or entry["nll_mean"] is None:
        return "pending"
    sd = entry["nll_sd"] if entry["nll_sd"] is not None else 0.0
    return f"{entry['nll_mean']:.4f} ({sd:.4f})"


def fmt_delta(target: dict[str, Any] | None, base: dict[str, Any] | None) -> str:
    if not target or not base:
        return "pending"
    if target["nll_mean"] is None or base["nll_mean"] is None:
        return "pending"
    return f"{target['nll_mean'] - base['nll_mean']:+.4f}"


def fmt_wf(entry: dict[str, Any] | None) -> str:
    if not entry or entry["wf_mean"] is None:
        return "pending"
    return f"{entry['wf_mean'] * 100:.1f}% ({entry['wf_kind']})"


def fmt_edge(entry: dict[str, Any] | None) -> str:
    if not entry or entry["edge_mean"] is None:
        return "pending"
    return f"{entry['edge_mean']:.4f}"


def row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def header(labels: list[str]) -> list[str]:
    return [row(labels), row(["---"] * len(labels))]


def nll_table(data: dict[int, dict[str, Any]]) -> list[str]:
    lines = ["", "## NLL by size", ""]
    lines += header(["n"] + [LABEL[key] for key in NLL_KEYS])
    for n in SIZES:
        cells = data.get(n, {})
        lines.append(row([str(n)] + [fmt_nll(cells.get(key)) for key in NLL_KEYS]))
    return lines


def shift_table(data: dict[int, dict[str, Any]]) -> list[str]:
    lines = ["", "## Shift by size", ""]
    lines += header(["n"] + [lbl for _, _, lbl in SHIFTS])
    for n in SIZES:
        cells = data.get(n, {})
        values = [str(n)]
        for target, base, _ in SHIFTS:
            values.append(fmt_delta(cells.get(target), cells.get(base)))
        lines.append(row(values))
    return lines


def wf_edge_table(data: dict[int, dict[str, Any]]) -> list[str]:
    lines = ["", "## WF and edge accuracy by size", ""]
    lines += header(["n", "lay WF", "mix WF", "lay edge acc (ID)", "mix edge acc (ID)"])
    for n in SIZES:
        cells = data.get(n, {})
        lines.append(row([
            str(n),
            fmt_wf(cells.get("lay_id")),
            fmt_wf(cells.get("mix_id")),
            fmt_edge(cells.get("lay_id")),
            fmt_edge(cells.get("mix_id")),
        ]))
    return lines


def parse_seeds(value: str) -> list[int]:
    try:
        return [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad seed list: {value}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    ap.add_argument("--out", default="runs/sweep_summary.md")
    args = ap.parse_args()

    data: dict[int, dict[str, Any]] = {}
    raw: dict[str, Any] = {}
    incomplete: list[str] = []

    for n in SIZES:
        data[n] = {}
        raw[str(n)] = {}
        for key, suffix, token, _ in CELLS:
            prefix = f"c_s{n}_{suffix}"
            tokens_dir = f"data/tok_s{n}_{token}"
            try:
                rep = summarize(args.runs, prefix, n, tokens_dir,
                                [CONFIG], BASELINE, args.seeds)
            except Exception as exc:  # noqa: BLE001 - pending cells must not crash
                data[n][key] = None
                raw[str(n)][key] = {"status": "pending", "error": str(exc)}
                incomplete.append(f"n{n}/{key} (0)")
                continue
            entry = extract(rep)
            data[n][key] = entry
            raw[str(n)][key] = rep
            if entry["n_seeds"] < 3:
                incomplete.append(f"n{n}/{key} ({entry['n_seeds']})")

    lines = ["# Milestone sweep summary", "",
             f"runs={args.runs}, seeds={args.seeds}, config={CONFIG}"]
    lines += nll_table(data)
    lines += shift_table(data)
    lines += wf_edge_table(data)
    text = "\n".join(lines) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(raw, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")

    print(text, end="")
    if incomplete:
        print("Incomplete (< 3 seeds): " + ", ".join(incomplete))
    else:
        print("Incomplete (< 3 seeds): none")


if __name__ == "__main__":
    main()
