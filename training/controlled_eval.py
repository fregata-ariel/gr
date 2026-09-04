"""
B: controlled evaluation on the untouched test split — cfg_reducer only,
no torch (docs/design/controlled_eval.md).

    uv run python -m training.controlled_eval --runs runs --prefix b_ --size 24 \
        --tokens data/tokens_b_n24 --configs base,mask,ptr --baseline base \
        --seeds 0,1,2 --out runs/b_summary_n24.json

Run directories are <prefix><config>_n<size>_s<seed> (e.g. b_mask_n24_s1)
and carry test_scores.jsonl (token_nll trace + ref_pos / ref_k /
ref_correct from train_ar.score_rows) and eval.json (eval_samples on the
diagnostic samples). Statistics follow the external review 2026-09-04:

  paired ΔNLL      same test samples, config vs baseline, per seed, with a
                   percentile-bootstrap CI over samples
  Wilson interval  for well-formed rates (WF kind labelled per config)
  NLL-by-k         with counts, bootstrap CI, and two frequency baselines:
                   unigram -log p_train(k) and a conditional-frequency
                   model P(k - klast | n_legal) fitted on train
  edge accuracy    argmax reference == true reference, overall and by k
  matched strata   NLL by k within fixed n_legal (candidate count), so the
                   uniform-over-candidates cost is constant within a row
"""

from __future__ import annotations
import argparse
import json
import random
from math import log, sqrt
from pathlib import Path
from statistics import mean, pstdev

from training import grammar_mask
from training.data_utils import read_jsonl

WF_KIND = {"base": "raw", "mask": "reference-constrained",
           "ptr": "reference-constrained", "ptr_depth": "reference-constrained"}
MIN_COUNT = 10


# ── statistics ───────────────────────────────

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_delta(scores: list[dict], baseline: list[dict],
                 key: str = "nll_per_token") -> dict:
    """Per-sample difference config - baseline on shared sample_ids."""
    base = {r["sample_id"]: r[key] for r in baseline}
    deltas = [r[key] - base[r["sample_id"]] for r in scores
              if r["sample_id"] in base]
    if not deltas:
        raise ValueError("no shared sample_ids between the two score files")
    lo, hi = bootstrap_ci(deltas)
    return {"n": len(deltas), "mean_delta": mean(deltas), "ci95": [lo, hi],
            "frac_improved": sum(d < 0 for d in deltas) / len(deltas)}


# ── REF-level tables ─────────────────────────

def ref_records(token_rows: dict[str, list[int]], scores: list[dict],
                vocab: dict[str, int], max_k: int) -> list[dict]:
    """One record per scored REF target: k, nll, n_legal, rel (= k - klast),
    correct (edge accuracy, when the score file carries it)."""
    names = {i: t for t, i in vocab.items()}
    out = []
    for s in scores:
        tokens = token_rows[s["sample_id"]]
        ctx = grammar_mask.pointer_context(tokens, vocab)
        correct = dict(zip(s.get("ref_pos", []), s.get("ref_correct", [])))
        for t, (tok, nll) in enumerate(zip(tokens[1:], s["token_nll"])):
            name = names[tok]
            if not name.startswith("REF_"):
                continue
            k = int(name[4:])
            klast = ctx["klast"][t]
            n_legal = max(0, min(max_k, ctx["lpos"][t] - 1) - klast)
            out.append({"k": k, "nll": nll, "n_legal": n_legal,
                        "rel": k - klast, "correct": correct.get(t)})
    return out


def by_k(records: list[dict], min_count: int = MIN_COUNT) -> dict[int, dict]:
    buckets: dict[int, list[float]] = {}
    for r in records:
        buckets.setdefault(r["k"], []).append(r["nll"])
    table = {}
    for k, v in sorted(buckets.items()):
        entry = {"n": len(v), "mean": mean(v)}
        if len(v) >= min_count:
            entry["ci95"] = list(bootstrap_ci(v))
        table[k] = entry
    return table


def macro_micro(table: dict[int, dict], min_count: int = MIN_COUNT) -> dict:
    """k-balanced (macro) vs frequency-weighted (micro) REF NLL."""
    rows = [(v["n"], v["mean"]) for v in table.values() if v["n"] >= min_count]
    if not rows:
        return {"macro": None, "micro": None}
    return {"macro": mean(m for _, m in rows),
            "micro": sum(n * m for n, m in rows) / sum(n for n, _ in rows)}


def edge_accuracy(records: list[dict]) -> dict:
    scored = [r for r in records if r["correct"] is not None]
    if not scored:
        return {"overall": None, "n": 0, "by_k": {}}
    per_k: dict[int, list[int]] = {}
    for r in scored:
        per_k.setdefault(r["k"], []).append(int(r["correct"]))
    return {
        "overall": mean(r["correct"] for r in scored), "n": len(scored),
        "ci95": list(wilson(sum(r["correct"] for r in scored), len(scored))),
        "by_k": {k: {"n": len(v), "acc": mean(v)} for k, v in sorted(per_k.items())},
    }


def within_n_legal(records: list[dict], min_count: int = MIN_COUNT
                   ) -> dict[int, dict[int, dict]]:
    """Matched strata: NLL by k within a fixed number of legal candidates."""
    strata: dict[int, dict[int, list[float]]] = {}
    for r in records:
        strata.setdefault(r["n_legal"], {}).setdefault(r["k"], []).append(r["nll"])
    return {
        c: {k: {"n": len(v), "mean": mean(v)}
            for k, v in sorted(ks.items()) if len(v) >= min_count}
        for c, ks in sorted(strata.items())
    }


# ── frequency baselines (fitted on train) ────

class FrequencyBaselines:
    """Unigram p(k) and conditional P(rel | n_legal), add-alpha smoothed,
    from the training split's REF tokens."""

    def __init__(self, train_rows: list[dict], vocab: dict[str, int],
                 max_k: int, alpha: float = 0.5) -> None:
        self.alpha = alpha
        recs = ref_records({r["sample_id"]: r["tokens"] for r in train_rows},
                           [{"sample_id": r["sample_id"],
                             "token_nll": [0.0] * (len(r["tokens"]) - 1)}
                            for r in train_rows], vocab, max_k)
        self.total = len(recs)
        self.count_k: dict[int, int] = {}
        self.count_rel: dict[int, dict[int, int]] = {}
        for r in recs:
            self.count_k[r["k"]] = self.count_k.get(r["k"], 0) + 1
            row = self.count_rel.setdefault(r["n_legal"], {})
            row[r["rel"]] = row.get(r["rel"], 0) + 1
        self.max_k = max_k

    def unigram_nll(self, k: int) -> float:
        return -log((self.count_k.get(k, 0) + self.alpha)
                    / (self.total + self.alpha * self.max_k))

    def conditional_nll(self, rel: int, n_legal: int) -> float:
        row = self.count_rel.get(n_legal, {})
        support = max(n_legal, 1)
        return -log((row.get(rel, 0) + self.alpha)
                    / (sum(row.values()) + self.alpha * support))

    def tables(self, records: list[dict], min_count: int = MIN_COUNT) -> dict:
        uni: dict[int, list[float]] = {}
        cond: dict[int, list[float]] = {}
        for r in records:
            uni.setdefault(r["k"], []).append(self.unigram_nll(r["k"]))
            cond.setdefault(r["k"], []).append(
                self.conditional_nll(r["rel"], r["n_legal"]))
        return {
            "unigram_nll_by_k": {k: mean(v) for k, v in sorted(uni.items())
                                 if len(v) >= min_count},
            "conditional_nll_by_k": {k: mean(v) for k, v in sorted(cond.items())
                                     if len(v) >= min_count},
        }


# ── per-run and summary ──────────────────────

def run_dir_name(prefix: str, config: str, size: int, seed: int) -> str:
    return f"{prefix}{config}_n{size}_s{seed}"


def evaluate_run(run_dir: Path, token_rows: dict[str, list[int]],
                 vocab: dict[str, int], max_k: int,
                 freq: FrequencyBaselines) -> dict:
    scores = read_jsonl(run_dir / "test_scores.jsonl")
    recs = ref_records(token_rows, scores, vocab, max_k)
    table = by_k(recs)
    out = {
        "run": run_dir.name,
        "test_nll_per_token": (sum(r["nll"] for r in scores)
                               / sum(r["n_tokens"] for r in scores)),
        "test_nll_per_sample_mean": mean(r["nll_per_token"] for r in scores),
        "ref_nll_by_k": table,
        "ref_nll_macro_micro": macro_micro(table),
        "edge_accuracy": edge_accuracy(recs),
        "within_n_legal": within_n_legal(recs),
        "frequency_baselines": freq.tables(recs),
    }
    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        out["wf"] = {"rate": ev["well_formed_rate"], "n": ev["total"],
                     "ci95": list(wilson(ev["well_formed"], ev["total"])),
                     "violations": ev["violations"]}
    return out


def summarize(runs_root: str | Path, prefix: str, size: int,
              tokens_dir: str | Path, configs: list[str], baseline: str,
              seeds: list[int]) -> dict:
    runs_root, tokens_dir = Path(runs_root), Path(tokens_dir)
    vocab = json.loads((tokens_dir / "vocab.json").read_text(encoding="utf-8"))
    max_k = max(int(t[4:]) for t in vocab if t.startswith("REF_"))
    token_rows = {r["sample_id"]: r["tokens"]
                  for r in read_jsonl(tokens_dir / "test.jsonl")}
    freq = FrequencyBaselines(read_jsonl(tokens_dir / "train.jsonl"), vocab, max_k)

    per_run: dict[str, dict[int, dict]] = {}
    scores_cache: dict[tuple[str, int], list[dict]] = {}
    for config in configs:
        for seed in seeds:
            run_dir = runs_root / run_dir_name(prefix, config, size, seed)
            if not (run_dir / "test_scores.jsonl").exists():
                continue
            per_run.setdefault(config, {})[seed] = evaluate_run(
                run_dir, token_rows, vocab, max_k, freq)
            scores_cache[(config, seed)] = read_jsonl(run_dir / "test_scores.jsonl")

    summary: dict[str, dict] = {}
    for config, by_seed in per_run.items():
        nlls = [r["test_nll_per_token"] for r in by_seed.values()]
        accs = [r["edge_accuracy"]["overall"] for r in by_seed.values()
                if r["edge_accuracy"]["overall"] is not None]
        entry = {
            "seeds": sorted(by_seed),
            "wf_kind": WF_KIND.get(config, "unknown"),
            "test_nll_per_token": {"mean": mean(nlls), "sd": pstdev(nlls) if len(nlls) > 1 else 0.0,
                                   "per_seed": nlls},
            "edge_accuracy": {"mean": mean(accs) if accs else None, "per_seed": accs},
        }
        wfs = [r["wf"] for r in by_seed.values() if "wf" in r]
        if wfs:
            entry["wf"] = {"mean": mean(w["rate"] for w in wfs),
                           "per_seed": [w["rate"] for w in wfs],
                           "ci95_per_seed": [w["ci95"] for w in wfs]}
        if config != baseline and baseline in per_run:
            deltas = {}
            for seed in by_seed:
                if (baseline, seed) in scores_cache:
                    deltas[seed] = paired_delta(scores_cache[(config, seed)],
                                                scores_cache[(baseline, seed)])
            if deltas:
                means = [d["mean_delta"] for d in deltas.values()]
                entry["paired_delta_vs_baseline"] = {
                    "per_seed": deltas, "mean": mean(means),
                    "all_same_sign": all(m < 0 for m in means) or all(m > 0 for m in means),
                }
        summary[config] = entry

    return {"size": size, "prefix": prefix, "baseline": baseline,
            "max_k": max_k, "n_test": len(token_rows),
            "summary": summary, "runs": per_run}


def print_summary(rep: dict) -> None:
    print(f"# n{rep['size']}  test n={rep['n_test']}  baseline={rep['baseline']}")
    for config, e in rep["summary"].items():
        nll = e["test_nll_per_token"]
        line = f"{config:10s} test NLL {nll['mean']:.4f} ±{nll['sd']:.4f} (seeds {e['seeds']})"
        if e["edge_accuracy"]["mean"] is not None:
            line += f"  edge acc {e['edge_accuracy']['mean']:.3f}"
        if "wf" in e:
            line += f"  {e['wf_kind']} WF {e['wf']['mean']*100:.1f}%"
        pd = e.get("paired_delta_vs_baseline")
        if pd:
            line += f"  paired ΔNLL {pd['mean']:+.4f} (same sign: {pd['all_same_sign']})"
        print(line)
    for config, by_seed in rep["runs"].items():
        first = by_seed[min(by_seed)]
        fb = first["frequency_baselines"]
        print(f"  {config} seed {min(by_seed)} REF NLL by k (n | model | unigram | conditional):")
        for k, v in first["ref_nll_by_k"].items():
            if v["n"] >= MIN_COUNT:
                print(f"    k{k}: n={v['n']:4d}  {v['mean']:.2f}  "
                      f"{fb['unigram_nll_by_k'].get(k, float('nan')):.2f}  "
                      f"{fb['conditional_nll_by_k'].get(k, float('nan')):.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training.controlled_eval")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--prefix", default="b_")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--configs", default="base,mask,ptr")
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rep = summarize(args.runs, args.prefix, args.size, args.tokens,
                    args.configs.split(","), args.baseline,
                    [int(s) for s in args.seeds.split(",")])
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    print_summary(rep)


if __name__ == "__main__":
    main()
