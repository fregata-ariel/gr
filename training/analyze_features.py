"""
Explore which structure features separate hard from easy samples
(docs/design/structure_sweep.md, stage 2) — stdlib only.

    uv run python -m training.analyze_features \
        --table runs/sweep_n24/features.jsonl [--table ...] \
        --target nll_per_token [--group n_motifs]

For every numeric feature: Spearman rank correlation with the target,
and the AUC of the feature as a classifier of the hard half (top 50%
target) vs the easy half. With several tables the analysis is run per
table (within-node-count) and pooled, so node count can be separated
from structure.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from training.data_utils import read_jsonl

NON_FEATURES = {"sample_id", "seed", "nll", "nll_per_token", "acc",
                "n_tokens"}


def _is_token_class(name: str) -> bool:
    """Columns derived from the score itself (nll_*_mean, n_*_tokens):
    they say WHICH tokens are hard, not which structure is hard."""
    return name.startswith("nll_") or (
        name.startswith("n_") and name.endswith("_tokens"))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def auc(feature: list[float], is_hard: list[bool]) -> float | None:
    """P(feature[hard] > feature[easy]) with ties counted 0.5."""
    hard = [f for f, h in zip(feature, is_hard) if h]
    easy = [f for f, h in zip(feature, is_hard) if not h]
    if not hard or not easy:
        return None
    ranks = _ranks(feature)
    rank_sum = sum(r for r, h in zip(ranks, is_hard) if h)
    n_h, n_e = len(hard), len(easy)
    return (rank_sum - n_h * (n_h + 1) / 2) / (n_h * n_e)


def analyze(rows: list[dict], target: str,
            token_class: bool = False) -> list[dict]:
    """Rank features by separability. token_class=False -> structural
    features only; True -> the score-derived token-class columns."""
    y = [r[target] for r in rows]
    median = sorted(y)[len(y) // 2]
    is_hard = [v > median for v in y]

    features = sorted(
        k for k in rows[0]
        if k not in NON_FEATURES and k != target
        and _is_token_class(k) == token_class
        and all(isinstance(r.get(k), (int, float)) for r in rows)
    )
    results = []
    for f in features:
        x = [float(r[f]) for r in rows]
        results.append({
            "feature": f,
            "spearman": spearman(x, y),
            "auc_hard": auc(x, is_hard),
            "n": len(rows),
        })
    results.sort(key=lambda r: -abs(r["auc_hard"] - 0.5)
                 if r["auc_hard"] is not None else 0)
    return results


def _print(title: str, results: list[dict], top: int) -> None:
    print(f"\n## {title}  (n={results[0]['n'] if results else 0})")
    if not results:
        return
    print(f"{'feature':20s} {'spearman':>9s} {'AUC(hard)':>10s}")
    for r in results[:top]:
        sp = f"{r['spearman']:+.3f}" if r["spearman"] is not None else "   n/a"
        au = f"{r['auc_hard']:.3f}" if r["auc_hard"] is not None else "  n/a"
        print(f"{r['feature']:20s} {sp:>9s} {au:>10s}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m training.analyze_features")
    parser.add_argument("--table", action="append", required=True,
                        help="features.jsonl (repeatable)")
    parser.add_argument("--target", default="nll_per_token")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args(argv)

    pooled: list[dict] = []
    for path in args.table:
        rows = read_jsonl(path)
        pooled.extend(rows)
        if len(args.table) > 1:
            _print(f"{Path(path).parent.name} — structure",
                   analyze(rows, args.target), args.top)
    label = "pooled" if len(args.table) > 1 else Path(args.table[0]).parent.name
    _print(f"{label} — structure", analyze(pooled, args.target), args.top)
    _print(f"{label} — token class (score-derived)",
           analyze(pooled, args.target, token_class=True), args.top)


if __name__ == "__main__":
    main()
