"""
Local step: evaluate generated token streams (docs/design/ar_baseline.md).

Run from the repo root:

    uv run python -m training.eval_samples --samples run1_samples.json \
        --vocab data/tokens1/vocab.json --train-tokens data/tokens1/train.jsonl

Metrics: well-formed rate (strict grammar via model_input.detokenize),
unique rate among well-formed sketches, and novelty against the
training split (memorization check).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from cfg_reducer import model_input
from training.data_utils import read_jsonl


def evaluate(
    streams: list[list[int]],
    vocab: dict[str, int],
    train_streams: list[list[int]],
) -> dict:
    train_sketches = {
        model_input.detokenize(s, vocab) for s in train_streams
    }

    well_formed: list = []
    for stream in streams:
        try:
            well_formed.append(model_input.detokenize(stream, vocab))
        except ValueError:
            pass

    unique = set(well_formed)
    novel = unique - train_sketches

    total = len(streams)
    return {
        "total": total,
        "well_formed": len(well_formed),
        "well_formed_rate": len(well_formed) / total if total else 0.0,
        "unique_sketches": len(unique),
        "unique_rate": (
            len(unique) / len(well_formed) if well_formed else 0.0
        ),
        "novel_sketches": len(novel),
        "novelty_rate": len(novel) / len(unique) if unique else 0.0,
        "avg_stream_len": (
            sum(len(s) for s in streams) / total if total else 0.0
        ),
    }


def _load_streams(path: str | Path) -> list[list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["samples"] if isinstance(payload, dict) else payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m training.eval_samples",
        description="Evaluate generated token streams.",
    )
    parser.add_argument("--samples", required=True,
                        help="samples.json from train_ar.py (or a JSON list)")
    parser.add_argument("--vocab", required=True, help="vocab.json")
    parser.add_argument("--train-tokens", required=True,
                        help="train.jsonl used for the novelty check")
    parser.add_argument("--out", default=None,
                        help="optional path for the JSON report")
    args = parser.parse_args(argv)

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    report = evaluate(
        _load_streams(args.samples),
        vocab,
        [row["tokens"] for row in read_jsonl(args.train_tokens)],
    )

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
