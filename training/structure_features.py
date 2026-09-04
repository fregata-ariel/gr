"""
Per-sample structure features for the structure-sweep analysis
(docs/design/structure_sweep.md) — cfg_reducer only, no torch.

Run from the repo root:

    uv run python -m training.structure_features \
        --dataset data/ds_sweep_n24 --tokens data/tokens_sweep_n24 \
        --scores runs/sweep_n24/val_scores.jsonl --out runs/sweep_n24/features.jsonl

Joins structural features of each val sample (from the canonical
MetaGraph) with the teacher-forced scores written by train_ar.py,
including per-token-class NLL (KIND / REF / LOOP / EOS).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean

from cfg_reducer import MetaGraph, store
from training.data_utils import read_jsonl, write_jsonl


# ── structural features ──────────────────────

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


# ── per-token-class NLL ──────────────────────

def _token_class(name: str) -> str:
    if name.startswith("KIND_"):
        return "kind"
    if name.startswith("REF_"):
        return "ref"
    if name.startswith("LOOP_"):
        return "loop"
    return name.lower()          # eos / bos / pad


def token_class_nll(tokens: list[int], token_nll: list[float],
                    vocab: dict[str, int]) -> dict:
    """Mean NLL per token class; token_nll aligns with tokens[1:]."""
    names = {i: t for t, i in vocab.items()}
    buckets: dict[str, list[float]] = {}
    for token, nll in zip(tokens[1:], token_nll):
        buckets.setdefault(_token_class(names[token]), []).append(nll)
    out = {}
    for cls in ("kind", "ref", "loop", "eos"):
        vals = buckets.get(cls, [])
        out[f"nll_{cls}_mean"] = mean(vals) if vals else None
        out[f"n_{cls}_tokens"] = len(vals)
    return out


# ── table build ──────────────────────────────

def build_table(dataset_dir: str | Path, tokens_dir: str | Path,
                scores_path: str | Path, split: str = "val") -> list[dict]:
    dataset_dir, tokens_dir = Path(dataset_dir), Path(tokens_dir)
    vocab = json.loads((tokens_dir / "vocab.json").read_text(encoding="utf-8"))
    token_rows = {r["sample_id"]: r for r in read_jsonl(tokens_dir / f"{split}.jsonl")}

    table = []
    for score in read_jsonl(scores_path):
        sid = score["sample_id"]
        mg = store.load_sample(dataset_dir / split / f"{sid}.json")
        row = {"sample_id": sid, "seed": score.get("seed")}
        row.update(features_for(mg))
        row.update({k: score[k] for k in ("n_tokens", "nll", "nll_per_token", "acc")})
        row.update(token_class_nll(token_rows[sid]["tokens"],
                                   score["token_nll"], vocab))
        table.append(row)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m training.structure_features",
        description="Join structural features with per-sample scores.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    table = build_table(args.dataset, args.tokens, args.scores, args.split)
    write_jsonl(args.out, table)
    print(f"wrote {len(table)} rows x {len(table[0]) if table else 0} columns to {args.out}")


if __name__ == "__main__":
    main()
