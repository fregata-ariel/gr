"""
Evaluation-axes report per run (docs/design/eval_axes.md) — cfg_reducer
only, no torch.

    uv run python -m training.eval_axes report --run runs/sweep_n24 \
        --dataset data/ds_sweep_n24 --tokens data/tokens_sweep_n24
    uv run python -m training.eval_axes compare runs/a/axes.json runs/b/axes.json

Primary axes are what a representation change should improve; canary
axes are reference values that should stay flat / keep their sign.
A flagged delta means 要検証 (record, then verify) — never a verdict.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean

from training.analyze_features import spearman
from training.data_utils import read_jsonl, write_jsonl
from training.structure_features import build_table

CANARY_FLAT = ["n_tokens", "max_out_degree", "max_in_degree",
               "max_scc_size", "mean_scc_size", "top_width"]
CANARY_SIGN = ["n_loops", "max_depth", "n_back_edges"]
PRIMARY_FEATURES = ["mean_offset", "max_offset", "max_width"]
RHO_THRESHOLD = 0.2
NLL_THRESHOLD = 0.1


# ── axis computations ────────────────────────

def ref_nll_by_k(token_rows: dict[str, list[int]], scores: list[dict],
                 vocab: dict[str, int], min_count: int = 10) -> dict[int, float]:
    names = {i: t for t, i in vocab.items()}
    buckets: dict[int, list[float]] = {}
    for s in scores:
        tokens = token_rows[s["sample_id"]]
        for tok, nll in zip(tokens[1:], s["token_nll"]):
            name = names[tok]
            if name.startswith("REF_"):
                buckets.setdefault(int(name[4:]), []).append(nll)
    return {k: mean(v) for k, v in sorted(buckets.items()) if len(v) >= min_count}


def feature_rho(features: list[dict], names: list[str],
                target: str = "nll_per_token") -> dict[str, float | None]:
    y = [r[target] for r in features]
    return {f: spearman([float(r[f]) for r in features], y) for f in names}


def nll_by_offset_bin(features: list[dict], bins: int = 3) -> list[dict]:
    rows = sorted(features, key=lambda r: r["mean_offset"])
    size = max(1, len(rows) // bins)
    out = []
    for i in range(bins):
        chunk = rows[i * size:(i + 1) * size] if i < bins - 1 else rows[i * size:]
        if chunk:
            out.append({
                "mean_offset_range": [round(chunk[0]["mean_offset"], 2),
                                      round(chunk[-1]["mean_offset"], 2)],
                "mean_nll_per_token": mean(r["nll_per_token"] for r in chunk),
                "n": len(chunk),
            })
    return out


def _class_mean(features: list[dict], key: str) -> float | None:
    vals = [r[key] for r in features if r.get(key) is not None]
    return mean(vals) if vals else None


# ── report ───────────────────────────────────

def build_report(run_dir: str | Path, dataset_dir: str | Path,
                 tokens_dir: str | Path) -> dict:
    run_dir, tokens_dir = Path(run_dir), Path(tokens_dir)
    features_path = run_dir / "features.jsonl"
    if not features_path.exists():
        write_jsonl(features_path, build_table(
            dataset_dir, tokens_dir, run_dir / "val_scores.jsonl"))
    features = read_jsonl(features_path)

    vocab = json.loads((tokens_dir / "vocab.json").read_text(encoding="utf-8"))
    token_rows = {r["sample_id"]: r["tokens"]
                  for r in read_jsonl(tokens_dir / "val.jsonl")}
    scores = read_jsonl(run_dir / "val_scores.jsonl")

    eval_path = run_dir / "eval.json"
    unconstrained = json.loads(eval_path.read_text()) if eval_path.exists() else None

    return {
        "run": run_dir.name,
        "n_val": len(features),
        "primary": {
            "ref_nll_by_k": ref_nll_by_k(token_rows, scores, vocab),
            "nll_by_offset_bin": nll_by_offset_bin(features),
            "feature_rho": feature_rho(features, PRIMARY_FEATURES),
            "val_nll_per_token": mean(r["nll_per_token"] for r in features),
            "unconstrained": None if unconstrained is None else {
                k: unconstrained[k] for k in
                ("well_formed_rate", "unique_rate", "novelty_rate",
                 "avg_stream_len", "violations")
            },
        },
        "canary_flat": feature_rho(features, CANARY_FLAT),
        "canary_sign": feature_rho(features, CANARY_SIGN),
        "canary_token_class": {
            k: _class_mean(features, k)
            for k in ("nll_kind_mean", "nll_loop_mean", "nll_eos_mean")
        },
        "invariants": {
            "n_entry_all_one": all(r["n_entry"] == 1 for r in features),
        },
    }


def compare(before: dict, after: dict) -> list[dict]:
    """Flag deltas worth recording and verifying (要検証)."""
    flags: list[dict] = []

    def rho_flags(section: str, sign_matters: bool) -> None:
        for f, b in before[section].items():
            a = after[section].get(f)
            if a is None or b is None:
                continue
            note = None
            if abs(a - b) > RHO_THRESHOLD:
                note = f"|Δρ| {abs(a - b):.2f} > {RHO_THRESHOLD}"
            if sign_matters and (a > 0) != (b > 0) and max(abs(a), abs(b)) >= 0.15:
                note = "sign flip"
            if note:
                flags.append({"axis": f"{section}.{f}", "before": b,
                              "after": a, "note": f"{note} — 要検証"})

    rho_flags("canary_flat", sign_matters=False)
    rho_flags("canary_sign", sign_matters=True)

    for k, b in before["canary_token_class"].items():
        a = after["canary_token_class"].get(k)
        if a is not None and b is not None and a - b > NLL_THRESHOLD:
            flags.append({"axis": f"canary_token_class.{k}", "before": b,
                          "after": a, "note": f"NLL +{a - b:.2f} — 要検証"})

    for k, ok in after["invariants"].items():
        if not ok:
            flags.append({"axis": f"invariants.{k}", "before": before["invariants"].get(k),
                          "after": ok, "note": "invariant violated — 要検証"})
    return flags


# ── printing ─────────────────────────────────

def print_report(rep: dict) -> None:
    p = rep["primary"]
    print(f"# {rep['run']}  (val n={rep['n_val']}, val NLL/token {p['val_nll_per_token']:.3f})")
    print("REF NLL by k: " + "  ".join(f"k{k}:{v:.2f}" for k, v in p["ref_nll_by_k"].items()))
    print("NLL by mean_offset bin: " + "  ".join(
        f"[{b['mean_offset_range'][0]}-{b['mean_offset_range'][1]}]:{b['mean_nll_per_token']:.3f}"
        for b in p["nll_by_offset_bin"]))
    if p["unconstrained"]:
        u = p["unconstrained"]
        print(f"unconstrained: WF {u['well_formed_rate']*100:.1f}%  unique {u['unique_rate']*100:.1f}%  "
              f"novelty {u['novelty_rate']*100:.1f}%  violations {u['violations']}")
    fmt = lambda d: "  ".join(f"{k}:{v:+.2f}" if v is not None else f"{k}:n/a" for k, v in d.items())
    print("primary ρ:  " + fmt(p["feature_rho"]))
    print("canary flat ρ (should stay ~0): " + fmt(rep["canary_flat"]))
    print("canary sign ρ (should stay <0): " + fmt(rep["canary_sign"]))
    print("canary token-class NLL: " + "  ".join(
        f"{k}:{v:.2f}" if v is not None else f"{k}:n/a" for k, v in rep["canary_token_class"].items()))
    print("invariants: " + ", ".join(f"{k}={v}" for k, v in rep["invariants"].items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training.eval_axes")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--tokens", required=True)
    r.add_argument("--out", default=None, help="default: <run>/axes.json")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = parser.parse_args(argv)

    if args.cmd == "report":
        rep = build_report(args.run, args.dataset, args.tokens)
        out = Path(args.out) if args.out else Path(args.run) / "axes.json"
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print_report(rep)
    else:
        before = json.loads(Path(args.before).read_text(encoding="utf-8"))
        after = json.loads(Path(args.after).read_text(encoding="utf-8"))
        flags = compare(before, after)
        print(f"# compare {before['run']} -> {after['run']}")
        pb, pa = before["primary"], after["primary"]
        print(f"val NLL/token: {pb['val_nll_per_token']:.3f} -> {pa['val_nll_per_token']:.3f}")
        ks = sorted(set(pb["ref_nll_by_k"]) & set(pa["ref_nll_by_k"]), key=int)
        print("REF NLL by k:  " + "  ".join(
            f"k{k}:{pb['ref_nll_by_k'][k]:.2f}->{pa['ref_nll_by_k'][k]:.2f}" for k in ks))
        if pb["unconstrained"] and pa["unconstrained"]:
            print(f"unconstrained WF: {pb['unconstrained']['well_formed_rate']*100:.1f}% -> "
                  f"{pa['unconstrained']['well_formed_rate']*100:.1f}%")
        print(f"flags ({len(flags)}):" if flags else "flags: none")
        for f in flags:
            print(f"  {f['axis']}: {f['before']} -> {f['after']}  {f['note']}")


if __name__ == "__main__":
    main()
