"""
Distribution fidelity of generated structures (docs/design/eval_axes.md).

    uv run python -m training.sketch_stats --vocab data/tokens_x/vocab.json \
        --reference data/tokens_x/val.jsonl --samples runs/r/samples_constrained.json

Computes sketch-level structure features (from detokenize, so it works
on generated streams without node ids) for a reference set and one or
more sample sets, and reports per-feature mean / std and a
Kolmogorov-Smirnov distance between each sample set and the reference.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from cfg_reducer import model_input
from training.data_utils import read_jsonl

FEATURES = ["n_motifs", "n_loops", "max_depth", "max_width",
            "mean_offset", "refs_per_motif", "merge_ratio", "n_tokens"]


def sketch_features(sketch, n_tokens: int) -> dict:
    """Structure features of a positional Sketch (nested tuples)."""
    motifs: list[str] = []
    offsets: list[int] = []
    widths: list[int] = []
    max_depth = 0

    def walk(level, depth):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        widths.append(len(level))
        for kind, offs, child in level:
            motifs.append(kind)
            offsets.extend(offs)
            if child is not None:
                walk(child, depth + 1)

    walk(sketch, 0)
    n = len(motifs)
    return {
        "n_motifs": n,
        "n_loops": motifs.count("loop"),
        "max_depth": max_depth,
        "max_width": max(widths) if widths else 0,
        "mean_offset": mean(offsets) if offsets else 0.0,
        "refs_per_motif": len(offsets) / n if n else 0.0,
        "merge_ratio": motifs.count("merge") / n if n else 0.0,
        "n_tokens": n_tokens,
    }


def features_of_streams(streams: list[list[int]], vocab: dict[str, int]) -> list[dict]:
    rows = []
    for stream in streams:
        try:
            sketch = model_input.detokenize(stream, vocab)
        except ValueError:
            continue                    # malformed streams are excluded
        rows.append(sketch_features(sketch, len(stream)))
    return rows


def ks_distance(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (0 = identical)."""
    if not a or not b:
        return 1.0
    xs = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)
    ia = ib = 0
    best = 0.0
    for x in xs:
        while ia < len(sa) and sa[ia] <= x:
            ia += 1
        while ib < len(sb) and sb[ib] <= x:
            ib += 1
        best = max(best, abs(ia / len(sa) - ib / len(sb)))
    return best


def compare(reference: list[dict], sample: list[dict]) -> dict:
    out = {}
    for f in FEATURES:
        ra = [float(r[f]) for r in reference]
        sa = [float(r[f]) for r in sample]
        out[f] = {
            "ref_mean": mean(ra), "ref_std": pstdev(ra),
            "sample_mean": mean(sa) if sa else None,
            "sample_std": pstdev(sa) if sa else None,
            "ks": ks_distance(ra, sa),
        }
    out["_n"] = {"reference": len(reference), "sample": len(sample)}
    return out


def _load_streams(path: str | Path) -> list[list[int]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [r["tokens"] for r in read_jsonl(p)]
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload["samples"] if isinstance(payload, dict) else payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training.sketch_stats")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--reference", required=True, help="val.jsonl")
    parser.add_argument("--samples", action="append", required=True,
                        help="samples json (repeatable, label with name=path)")
    args = parser.parse_args(argv)

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    reference = features_of_streams(_load_streams(args.reference), vocab)
    header = f"{'feature':15s} {'ref mean±sd':>16s}"
    labeled = []
    for spec in args.samples:
        name, _, path = spec.partition("=") if "=" in spec else (Path(spec).parent.name, "", spec)
        labeled.append((name, compare(reference, features_of_streams(_load_streams(path), vocab))))
        header += f" | {name + ' mean (KS)':>22s}"
    print(header)
    for f in FEATURES:
        row = f"{f:15s} {labeled[0][1][f]['ref_mean']:8.2f}±{labeled[0][1][f]['ref_std']:<6.2f}"
        for name, rep in labeled:
            m, ks = rep[f]["sample_mean"], rep[f]["ks"]
            row += f" | {m:12.2f}  ({ks:.3f}) " if m is not None else f" | {'n/a':>22s}"
        print(row)
    for name, rep in labeled:
        print(f"[{name}] n_reference={rep['_n']['reference']} n_sample={rep['_n']['sample']}  "
              f"mean KS={mean(rep[f]['ks'] for f in FEATURES):.3f}")


if __name__ == "__main__":
    main()
