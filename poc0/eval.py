from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cfg_reducer.linearize import decode

from poc0.constants import SAMPLING_SEED, SPLIT_SEED, TRAIN_SPLIT_FRACTION
from poc0.sample import InvalidSampleReason, SampleResult, restore_logits_fn_from_checkpoint, sample_tokens
from poc0.stats import normalized_histogram, stats_for_skeleton, total_variation_distance

_STAT_NAMES = (
    "n_motifs",
    "n_tokens",
    "depth",
    "max_width",
    "loop_nest",
    "kinds.entry",
    "kinds.linear",
    "kinds.merge",
    "kinds.loop",
)
_KIND_NAMES = ("entry", "linear", "merge", "loop")


@dataclass(frozen=True)
class _JsonlRecord:
    record_id: str
    tokens: list[str]
    stats: dict[str, object] | None


def _load_jsonl_records(jsonl_path: Path) -> list[_JsonlRecord]:
    records: list[_JsonlRecord] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL line {line_number} must contain an object record"
                )

            record_id = payload.get("id")
            tokens = payload.get("tokens")
            stats = payload.get("stats")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"JSONL line {line_number} is missing a string record id"
                )
            if not isinstance(tokens, list) or not all(
                isinstance(token, str) for token in tokens
            ):
                raise ValueError(
                    f"JSONL line {line_number} is missing a string token list"
                )
            if stats is not None and not isinstance(stats, dict):
                raise ValueError(f"JSONL line {line_number} stats must be an object")

            records.append(
                _JsonlRecord(
                    record_id=record_id,
                    tokens=tokens,
                    stats=stats,
                )
            )

    return sorted(records, key=lambda record: record.record_id)


def _split_train_records(
    records: list[_JsonlRecord],
    *,
    split_seed: int,
) -> list[_JsonlRecord]:
    shuffled = list(records)
    random.Random(split_seed).shuffle(shuffled)
    train_size = int(TRAIN_SPLIT_FRACTION * len(shuffled))
    return shuffled[:train_size]


def _load_manifest_train_ids(manifest_path: Path) -> tuple[str, ...]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest.json is missing splits")
    train_split = splits.get("train")
    if not isinstance(train_split, dict):
        raise ValueError("manifest.json is missing train split")
    record_ids = train_split.get("record_ids")
    if not isinstance(record_ids, list) or not all(isinstance(record_id, str) for record_id in record_ids):
        raise ValueError("manifest.json train split is missing record_ids")
    return tuple(record_ids)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer stat value, got {type(value).__name__}")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a numeric value, got {type(value).__name__}")
    return float(value)


def _flatten_stats(stats: Mapping[str, object]) -> dict[str, int]:
    kinds = stats.get("kinds")
    if kinds is not None and not isinstance(kinds, Mapping):
        raise ValueError("stats.kinds must be an object")

    flat = {
        "n_motifs": _as_int(stats["n_motifs"]),
        "n_tokens": _as_int(stats["n_tokens"]),
        "depth": _as_int(stats["depth"]),
        "max_width": _as_int(stats["max_width"]),
        "loop_nest": _as_int(stats["loop_nest"]),
    }
    for kind in _KIND_NAMES:
        flat[f"kinds.{kind}"] = 0 if kinds is None else _as_int(kinds.get(kind, 0))
    return flat


def _token_key(tokens: list[str]) -> str:
    return "\x1f".join(tokens)


def _resolve_train_records(
    *,
    jsonl_path: Path,
    manifest_path: Path | None,
) -> list[_JsonlRecord]:
    all_records = _load_jsonl_records(jsonl_path)
    if manifest_path is None:
        return _split_train_records(all_records, split_seed=SPLIT_SEED)

    record_by_id = {record.record_id: record for record in all_records}
    train_ids = _load_manifest_train_ids(manifest_path)
    missing = [record_id for record_id in train_ids if record_id not in record_by_id]
    if missing:
        raise ValueError(
            f"manifest references record ids missing from JSONL: {missing[:3]}"
        )
    return [record_by_id[record_id] for record_id in train_ids]


def _train_record_stats(record: _JsonlRecord) -> dict[str, int]:
    if record.stats is not None:
        return _flatten_stats(record.stats)
    return _flatten_stats(stats_for_skeleton(decode(record.tokens), record.tokens))


def _verify_train_stats_subset(train_records: list[_JsonlRecord]) -> None:
    checked = 0
    for record in train_records:
        if record.stats is None:
            continue
        derived = _flatten_stats(stats_for_skeleton(decode(record.tokens), record.tokens))
        recorded = _flatten_stats(record.stats)
        if derived != recorded:
            raise ValueError(
                f"JSONL stats do not match decoded Skeleton for record {record.record_id}"
            )
        checked += 1
        if checked >= min(10, len(train_records)):
            return


def _reason_key(reason: str | None) -> str:
    if reason is None:
        raise ValueError("invalid sample reason must not be None")
    return str(reason)


def _histogram_counts(values: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(values).items()))


def _build_histogram_report(
    *,
    train_values: Sequence[int],
    sample_values: Sequence[int],
) -> dict[str, object]:
    train_probs = normalized_histogram(train_values)
    sample_probs = normalized_histogram(sample_values)
    return {
        "train_counts": _histogram_counts(train_values),
        "sample_counts": _histogram_counts(sample_values),
        "train_probs": train_probs,
        "sample_probs": sample_probs,
        "tvd": total_variation_distance(train_probs, sample_probs),
    }


def evaluate_samples(
    *,
    samples: Sequence[SampleResult],
    jsonl_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    train_records = _resolve_train_records(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )
    _verify_train_stats_subset(train_records)

    train_strings = {_token_key(record.tokens) for record in train_records}
    train_flat_stats = [_train_record_stats(record) for record in train_records]

    invalid_reasons = {
        InvalidSampleReason.LENGTH_CAP.value: 0,
        InvalidSampleReason.NEGATIVE_DEPTH.value: 0,
        InvalidSampleReason.DECODE_ERROR.value: 0,
    }

    valid_strings: list[str] = []
    valid_flat_stats: list[dict[str, int]] = []

    for sample in samples:
        if not sample.success:
            invalid_reasons[_reason_key(sample.invalid_reason)] += 1
            continue

        try:
            skeleton = decode(sample.tokens)
        except ValueError:
            invalid_reasons[InvalidSampleReason.DECODE_ERROR.value] += 1
            continue

        valid_strings.append(_token_key(sample.tokens))
        valid_flat_stats.append(_flatten_stats(stats_for_skeleton(skeleton, sample.tokens)))

    valid_count = len(valid_strings)
    invalid_count = len(samples) - valid_count
    duplicate_of_train_count = sum(token_key in train_strings for token_key in valid_strings)
    novel_count = valid_count - duplicate_of_train_count
    unique_valid_count = len(set(valid_strings))
    repeated_valid_count = valid_count - unique_valid_count

    histograms: dict[str, dict[str, object]] = {}
    stat_tvd: dict[str, float] = {}
    for stat_name in _STAT_NAMES:
        train_values = [stats[stat_name] for stats in train_flat_stats]
        sample_values = [stats[stat_name] for stats in valid_flat_stats]
        histogram = _build_histogram_report(
            train_values=train_values,
            sample_values=sample_values,
        )
        histograms[stat_name] = histogram
        stat_tvd[stat_name] = _as_float(histogram["tvd"])

    return {
        "n_samples": len(samples),
        "train_record_count": len(train_records),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validity_rate": 0.0 if not samples else valid_count / len(samples),
        "invalid_reasons": invalid_reasons,
        "novelty_rate": 0.0 if valid_count == 0 else novel_count / valid_count,
        "duplicate_of_train_rate": (
            0.0 if valid_count == 0 else duplicate_of_train_count / valid_count
        ),
        "unique_valid_count": unique_valid_count,
        "repeated_valid_count": repeated_valid_count,
        "stat_tvd": stat_tvd,
        "histograms": histograms,
    }


def _plot_histogram_overlay(
    *,
    stat_name: str,
    histogram: Mapping[str, object],
    out_dir: Path,
) -> None:
    train_probs = histogram["train_probs"]
    sample_probs = histogram["sample_probs"]
    assert isinstance(train_probs, Mapping)
    assert isinstance(sample_probs, Mapping)

    bins = sorted(set(train_probs) | set(sample_probs))
    train_values = [_as_float(train_probs.get(bin_value, 0.0)) for bin_value in bins]
    sample_values = [_as_float(sample_probs.get(bin_value, 0.0)) for bin_value in bins]
    positions = list(range(len(bins)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([position - 0.2 for position in positions], train_values, width=0.4, label="train", alpha=0.7)
    ax.bar([position + 0.2 for position in positions], sample_values, width=0.4, label="sample", alpha=0.7)
    ax.set_title(stat_name)
    ax.set_xlabel("bin")
    ax.set_ylabel("probability")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(bin_value) for bin_value in bins])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stat_name}.png")
    plt.close(fig)


def write_eval_report(
    *,
    report: Mapping[str, object],
    out_dir: Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histograms = report["histograms"]
    assert isinstance(histograms, Mapping)
    for stat_name, histogram in histograms.items():
        assert isinstance(stat_name, str)
        assert isinstance(histogram, Mapping)
        _plot_histogram_overlay(
            stat_name=stat_name,
            histogram=cast(Mapping[str, object], histogram),
            out_dir=out_dir,
        )

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics_path


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    jsonl_path: Path,
    out_dir: Path,
    n_samples: int,
    temperature: float,
    seed: int,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    logits_fn, checkpoint_manifest = restore_logits_fn_from_checkpoint(checkpoint_path)
    rng = jax.random.key(seed)
    samples: list[SampleResult] = []
    for _ in range(n_samples):
        rng, sample_rng = jax.random.split(rng)
        samples.append(
            sample_tokens(
                logits_fn=logits_fn,
                temperature=temperature,
                rng=sample_rng,
            )
        )

    report = evaluate_samples(
        samples=samples,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )
    report["checkpoint"] = str(checkpoint_path)
    report["temperature"] = float(temperature)
    report["seed"] = int(seed)
    report["checkpoint_manifest"] = checkpoint_manifest
    write_eval_report(report=report, out_dir=out_dir)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m poc0.eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        jsonl_path=args.jsonl,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
