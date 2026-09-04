"""
Local step: canonical dataset directory -> token files for the AR
baseline (docs/design/ar_baseline.md).

Run from the repo root:

    uv run python -m training.prepare_tokens --dataset data/ds1 --out data/tokens1

Builds ONE vocabulary sized by the max backward offset across every
split (so train and val share token ids), then writes vocab.json,
meta.json and <split>.jsonl.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from cfg_reducer import MetaGraph, model_input, store
from training.data_utils import write_jsonl


def prepare(dataset_dir: str | Path, out_dir: str | Path) -> dict:
    dataset_path = Path(dataset_dir)
    manifest = json.loads(
        (dataset_path / "manifest.json").read_text(encoding="utf-8")
    )

    # Pass 1 — load every sample and find the global offset window.
    loaded: dict[str, list[tuple[dict, MetaGraph]]] = {}
    max_offset = 1
    for split_name, info in manifest["splits"].items():
        rows = []
        for entry in info["samples"]:
            mg = store.load_sample(
                dataset_path / split_name / f"{entry['sample_id']}.json"
            )
            max_offset = max(max_offset, model_input.max_offset_needed(mg))
            rows.append((entry, mg))
        loaded[split_name] = rows

    # Pass 2 — tokenize with the shared vocabulary.
    vocab = model_input.build_vocab(max_offset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_len = 0
    counts: dict[str, int] = {}
    for split_name, rows in loaded.items():
        records = []
        for entry, mg in rows:
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
    args = parser.parse_args(argv)

    meta = prepare(args.dataset, args.out)
    print(
        f"vocab window REF_1..REF_{meta['max_offset']}, "
        f"max stream length {meta['max_len']}, "
        f"splits {meta['splits']}"
    )


if __name__ == "__main__":
    main()
