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

from cfg_reducer import store
from cfg_reducer.structure_features import (
    features_for as features_for, _walk_levels as _walk_levels,
)
from training.data_utils import read_jsonl, write_jsonl


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
