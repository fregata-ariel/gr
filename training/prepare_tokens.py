"""
Local step: canonical dataset directory -> token files for the AR
baseline (docs/design/ar_baseline.md).

Run from the repo root:

    uv run python -m training.prepare_tokens --dataset data/ds1 --out data/tokens1

Builds ONE vocabulary sized by the max backward offset — by default
across every split (so train and val share token ids); with
--window-from train the window (and hence the model shape) is decided
by the training split alone and samples of other splits that need a
larger offset are excluded and listed in meta.json (external review
2026-09-04, 1-1 / B-1).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from cfg_reducer import MetaGraph, model_input, store
from training.data_utils import write_jsonl


def prepare(dataset_dir: str | Path, out_dir: str | Path,
            window_from: str | None = None) -> dict:
    dataset_path = Path(dataset_dir)
    manifest = json.loads(
        (dataset_path / "manifest.json").read_text(encoding="utf-8")
    )
    if window_from is not None and window_from not in manifest["splits"]:
        raise ValueError(f"unknown split {window_from!r} for --window-from")

    # Pass 1 — load every sample and find the offset window (all splits,
    # or only window_from).
    loaded: dict[str, list[tuple[dict, MetaGraph, int]]] = {}
    max_offset = 1
    for split_name, info in manifest["splits"].items():
        rows = []
        for entry in info["samples"]:
            mg = store.load_sample(
                dataset_path / split_name / f"{entry['sample_id']}.json"
            )
            needed = model_input.max_offset_needed(mg)
            if window_from is None or split_name == window_from:
                max_offset = max(max_offset, needed)
            rows.append((entry, mg, needed))
        loaded[split_name] = rows

    # Pass 2 — tokenize with the shared vocabulary.
    vocab = model_input.build_vocab(max_offset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_len = 0
    counts: dict[str, int] = {}
    excluded: dict[str, list[int]] = {}
    for split_name, rows in loaded.items():
        records = []
        for entry, mg, needed in rows:
            if needed > max_offset:
                excluded.setdefault(split_name, []).append(entry["seed"])
                continue
            tokens = model_input.tokenize(mg, vocab)
            max_len = max(max_len, len(tokens))
            records.append({
                "sample_id": entry["sample_id"],
                "seed": entry["seed"],
                "tokens": tokens,
            })
        write_jsonl(out / f"{split_name}.jsonl", records)
        counts[split_name] = len(records)

    meta = {"max_offset": max_offset, "max_len": max_len, "splits": counts}
    if window_from is not None:
        meta["window_from"] = window_from
        meta["excluded_over_window"] = {
            split: {"count": len(seeds), "seeds": seeds}
            for split, seeds in excluded.items()
        }
    (out / "vocab.json").write_text(
        json.dumps(vocab, indent=2), encoding="utf-8"
    )
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m training.prepare_tokens",
        description="Tokenize a canonical MetaGraph dataset for training.",
    )
    parser.add_argument("--dataset", required=True,
                        help="dataset directory (with manifest.json)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--window-from", default=None, metavar="SPLIT",
        help="size the REF window from this split only (e.g. train); "
             "other splits' samples needing a larger offset are excluded "
             "and listed in meta.json",
    )
    args = parser.parse_args(argv)

    meta = prepare(args.dataset, args.out, args.window_from)
    print(
        f"vocab window REF_1..REF_{meta['max_offset']}, "
        f"max stream length {meta['max_len']}, "
        f"splits {meta['splits']}"
    )
    if meta.get("excluded_over_window"):
        print("excluded over window: " + ", ".join(
            f"{k}={v['count']}" for k, v in meta["excluded_over_window"].items()))


if __name__ == "__main__":
    main()
