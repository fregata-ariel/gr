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
import hashlib
import json
from pathlib import Path

from cfg_reducer import MetaGraph, model_input, store
from training.data_utils import write_jsonl


def prepare(dataset_dir: str | Path, out_dir: str | Path,
            window_from: str | None = None, *,
            test_dataset: str | Path | None = None) -> dict:
    if test_dataset is not None:
        if window_from != "train":
            raise ValueError("--test-dataset requires --window-from train")
        return _prepare_cross_dataset(Path(dataset_dir), Path(test_dataset), Path(out_dir))
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


def _prepare_cross_dataset(source: Path, target: Path, out: Path) -> dict:
    """Build a source-shaped bundle with provenance and exclusion denominators."""
    source_bytes = (source / "manifest.json").read_bytes()
    target_bytes = (target / "manifest.json").read_bytes()
    source_manifest, target_manifest = json.loads(source_bytes), json.loads(target_bytes)
    sources = {split: hashlib.sha256(data).hexdigest() for split, data in
               (("train", source_bytes), ("val", source_bytes), ("test", target_bytes))}
    loaded = {}
    seen = set()
    for split, path, manifest in (("train", source, source_manifest),
                                  ("val", source, source_manifest),
                                  ("test", target, target_manifest)):
        rows = []
        for entry in manifest["splits"][split]["samples"]:
            sid = entry["sample_id"]
            if sid in seen:
                raise ValueError(f"duplicate sample_id: {sid}")
            seen.add(sid)
            mg = store.load_sample(path / split / f"{sid}.json")
            rows.append((entry, mg, model_input.max_offset_needed(mg)))
        loaded[split] = rows
    if not loaded["train"]:
        raise ValueError("source train must be nonempty")
    max_offset = max(1, max(row[2] for row in loaded["train"]))
    vocab = model_input.build_vocab(max_offset)
    max_len = max(len(model_input.tokenize(row[1], vocab)) for row in loaded["train"])
    capacity = 2 * max_len
    index = {"version": 1, "sources": sources,
             "selection": target_manifest.get("selection"), "samples": {}, "exclusions": []}
    counts = {}
    excluded_window = {}
    output = {}
    for split, rows in loaded.items():
        records = []
        stats = {"raw": len(rows), "retained": 0, "excluded": 0, "by_bucket": {}}
        for entry, mg, needed in rows:
            sid, bucket = entry["sample_id"], entry.get("bucket")
            group = stats["by_bucket"].setdefault(
                bucket if bucket is not None else "__unmeasured__",
                {"raw": 0, "retained": 0, "excluded": 0})
            group["raw"] += 1
            reason = None
            tokens = []
            if needed > max_offset:
                reason = "over_window"
                excluded_window.setdefault(split, []).append(entry["seed"])
            else:
                tokens = model_input.tokenize(mg, vocab)
                if split != "train" and len(tokens) > capacity:
                    reason, needed = "over_length", len(tokens)
            if reason:
                index["exclusions"].append({"sample_id": sid, "seed": entry["seed"],
                    "split": split, "reason": reason, "needed": needed, "bucket": bucket})
                stats["excluded"] += 1
                group["excluded"] += 1
            else:
                records.append({"sample_id": sid, "seed": entry["seed"], "tokens": tokens})
                index["samples"][sid] = {"split": split, "bucket": bucket,
                                            "realized": entry.get("realized")}
                stats["retained"] += 1
                group["retained"] += 1
        counts[split], output[split] = stats, records
    meta = {"max_offset": max_offset, "max_len": max_len, "max_len_source": "train",
            "sequence_capacity": capacity, "window_from": "train", "sources": sources,
            "splits": {s: c["retained"] for s, c in counts.items()}, "counts": counts,
            "excluded_over_window": {s: {"count": len(v), "seeds": v}
                                     for s, v in excluded_window.items()},
            "exclusions": index["exclusions"]}
    out.mkdir(parents=True, exist_ok=True)
    for split, records in output.items():
        write_jsonl(out / f"{split}.jsonl", records)
    for name, data in (("vocab", vocab), ("meta", meta), ("evaluation_index", index)):
        (out / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
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
    parser.add_argument("--test-dataset", metavar="DIR")
    args = parser.parse_args(argv)

    meta = prepare(args.dataset, args.out, args.window_from, test_dataset=args.test_dataset)
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
