from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import poc0.eval as eval_module
from poc0.eval import evaluate_samples, write_eval_report
from poc0.sample import SampleResult


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def _write_manifest(path: Path, train_record_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "record_ids": train_record_ids,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_eval_validity_uses_decode_and_sampler_termination(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "train-1",
                "tokens": ["ADD_ENTRY", "STOP"],
                "stats": {
                    "n_motifs": 1,
                    "kinds": {"entry": 1, "linear": 0, "merge": 0, "loop": 0},
                    "n_tokens": 2,
                    "depth": 1,
                    "max_width": 1,
                    "loop_nest": 0,
                },
            }
        ],
    )

    report = evaluate_samples(
        samples=[
            SampleResult(tokens=["ADD_ENTRY", "STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["OPEN", "STOP"], success=True, invalid_reason=None, depth=1),
            SampleResult(tokens=["OPEN"], success=False, invalid_reason="length_cap", depth=1),
        ],
        jsonl_path=jsonl_path,
    )

    assert report["valid_count"] == 1
    assert report["invalid_count"] == 2
    assert report["invalid_reasons"] == {
        "length_cap": 1,
        "negative_depth": 0,
        "decode_error": 1,
    }


def test_eval_novelty_counts_only_valid_samples(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "train-1",
                "tokens": ["ADD_ENTRY", "STOP"],
                "stats": {
                    "n_motifs": 1,
                    "kinds": {"entry": 1, "linear": 0, "merge": 0, "loop": 0},
                    "n_tokens": 2,
                    "depth": 1,
                    "max_width": 1,
                    "loop_nest": 0,
                },
            }
        ],
    )
    _write_manifest(manifest_path, ["train-1"])

    report = evaluate_samples(
        samples=[
            SampleResult(tokens=["ADD_ENTRY", "STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["ADD_LINEAR", "STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["OPEN"], success=False, invalid_reason="length_cap", depth=1),
        ],
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )

    assert report["valid_count"] == 2
    assert report["novelty_rate"] == 0.5
    assert report["duplicate_of_train_rate"] == 0.5
    assert report["unique_valid_count"] == 2
    assert report["repeated_valid_count"] == 0


def test_eval_tracks_empty_streams_separately_from_valid_and_invalid(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "train-1",
                "tokens": ["ADD_ENTRY", "STOP"],
                "stats": {
                    "n_motifs": 1,
                    "kinds": {"entry": 1, "linear": 0, "merge": 0, "loop": 0},
                    "n_tokens": 2,
                    "depth": 1,
                    "max_width": 1,
                    "loop_nest": 0,
                },
            }
        ],
    )
    _write_manifest(manifest_path, ["train-1"])

    report = evaluate_samples(
        samples=[
            SampleResult(tokens=["STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["ADD_ENTRY", "STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["ADD_LINEAR", "STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["OPEN"], success=False, invalid_reason="length_cap", depth=1),
        ],
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )

    assert report["n_samples"] == 4
    assert report["empty_stream_count"] == 1
    assert report["empty_stream_rate"] == 0.25
    assert report["valid_count"] == 2
    assert report["invalid_count"] == 1
    assert report["validity_rate"] == 0.5
    assert report["invalid_reasons"] == {
        "length_cap": 1,
        "negative_depth": 0,
        "decode_error": 0,
    }
    assert report["novelty_rate"] == 0.5
    assert report["duplicate_of_train_rate"] == 0.5
    assert report["unique_valid_count"] == 2
    assert report["repeated_valid_count"] == 0
    histograms = cast(dict[str, object], report["histograms"])
    n_motifs_histogram = cast(dict[str, object], histograms["n_motifs"])
    assert n_motifs_histogram["sample_counts"] == {1: 2}


def test_eval_report_contains_required_metrics(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    out_dir = tmp_path / "eval_out"
    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "train-1",
                "tokens": ["ADD_ENTRY", "STOP"],
                "stats": {
                    "n_motifs": 1,
                    "kinds": {"entry": 1, "linear": 0, "merge": 0, "loop": 0},
                    "n_tokens": 2,
                    "depth": 1,
                    "max_width": 1,
                    "loop_nest": 0,
                },
            }
        ],
    )

    report = evaluate_samples(
        samples=[
            SampleResult(tokens=["ADD_ENTRY", "STOP"], success=True, invalid_reason=None, depth=0),
            SampleResult(tokens=["ADD_LINEAR", "STOP"], success=True, invalid_reason=None, depth=0),
        ],
        jsonl_path=jsonl_path,
    )
    metrics_path = write_eval_report(report=report, out_dir=out_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics_path == out_dir / "metrics.json"
    assert "empty_stream_count" in metrics
    assert "empty_stream_rate" in metrics
    assert "validity_rate" in metrics
    assert "novelty_rate" in metrics
    assert "duplicate_of_train_rate" in metrics
    assert "stat_tvd" in metrics
    assert "histograms" in metrics
    assert (out_dir / "n_motifs.png").exists()
    assert (out_dir / "kinds.entry.png").exists()

    expected_stats = {
        "n_motifs",
        "n_tokens",
        "depth",
        "max_width",
        "loop_nest",
        "kinds.entry",
        "kinds.linear",
        "kinds.merge",
        "kinds.loop",
    }
    assert set(metrics["stat_tvd"]) == expected_stats


def test_eval_cli_dump_samples_uses_eval_classification_and_keeps_metrics_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    jsonl_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir_plain = tmp_path / "eval_plain"
    out_dir_dump = tmp_path / "eval_dump"
    # Nested nonexistent directory: the dump writer must create parents
    # (regression: first Colab run crashed with FileNotFoundError).
    dump_path = tmp_path / "nested" / "dump_dir" / "samples.jsonl"

    _write_jsonl(
        jsonl_path,
        [
            {
                "id": "train-1",
                "tokens": ["ADD_ENTRY", "STOP"],
                "stats": {
                    "n_motifs": 1,
                    "kinds": {"entry": 1, "linear": 0, "merge": 0, "loop": 0},
                    "n_tokens": 2,
                    "depth": 1,
                    "max_width": 1,
                    "loop_nest": 0,
                },
            },
            {
                "id": "train-2",
                "tokens": ["ADD_LINEAR", "STOP"],
                "stats": {
                    "n_motifs": 1,
                    "kinds": {"entry": 0, "linear": 1, "merge": 0, "loop": 0},
                    "n_tokens": 2,
                    "depth": 1,
                    "max_width": 1,
                    "loop_nest": 0,
                },
            },
        ],
    )
    _write_manifest(manifest_path, ["train-1"])

    scripted_samples = [
        SampleResult(tokens=["ADD_MERGE", "STOP"], success=True, invalid_reason=None, depth=0),
        SampleResult(tokens=["ADD_ENTRY", "STOP"], success=True, invalid_reason=None, depth=0),
        SampleResult(tokens=["STOP"], success=True, invalid_reason=None, depth=0),
        SampleResult(tokens=["OPEN"], success=False, invalid_reason="length_cap", depth=1),
        SampleResult(tokens=["CLOSE"], success=False, invalid_reason="negative_depth", depth=-1),
        SampleResult(tokens=["OPEN", "STOP"], success=True, invalid_reason=None, depth=1),
    ]

    sample_iter = iter(())

    def fake_restore_logits_fn_from_checkpoint(path: Path):
        assert path == checkpoint_path
        return (object(), {"step": 123})

    def fake_sample_tokens(*, logits_fn: object, temperature: float, rng: object) -> SampleResult:
        assert logits_fn is not None
        assert temperature == 1.0
        return next(sample_iter)

    monkeypatch.setattr(
        eval_module,
        "restore_logits_fn_from_checkpoint",
        fake_restore_logits_fn_from_checkpoint,
    )
    monkeypatch.setattr(eval_module, "sample_tokens", fake_sample_tokens)

    sample_iter = iter(scripted_samples)
    assert (
        eval_module.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--jsonl",
                str(jsonl_path),
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(out_dir_plain),
                "--n-samples",
                str(len(scripted_samples)),
            ]
        )
        == 0
    )

    metrics_plain = (out_dir_plain / "metrics.json").read_bytes()

    sample_iter = iter(scripted_samples)
    assert (
        eval_module.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--jsonl",
                str(jsonl_path),
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(out_dir_dump),
                "--n-samples",
                str(len(scripted_samples)),
                "--dump-samples",
                str(dump_path),
            ]
        )
        == 0
    )

    assert (out_dir_dump / "metrics.json").read_bytes() == metrics_plain
    dump_records = [
        json.loads(line)
        for line in dump_path.read_text(encoding="utf-8").splitlines()
    ]
    assert dump_records == [
        {
            "index": 0,
            "tokens": ["ADD_MERGE", "STOP"],
            "success": True,
            "invalid_reason": None,
            "decode_error": None,
            "empty_stream": False,
            "in_train": False,
        },
        {
            "index": 1,
            "tokens": ["ADD_ENTRY", "STOP"],
            "success": True,
            "invalid_reason": None,
            "decode_error": None,
            "empty_stream": False,
            "in_train": True,
        },
        {
            "index": 2,
            "tokens": ["STOP"],
            "success": True,
            "invalid_reason": None,
            "decode_error": None,
            "empty_stream": True,
            "in_train": None,
        },
        {
            "index": 3,
            "tokens": ["OPEN"],
            "success": False,
            "invalid_reason": "length_cap",
            "decode_error": None,
            "empty_stream": False,
            "in_train": None,
        },
        {
            "index": 4,
            "tokens": ["CLOSE"],
            "success": False,
            "invalid_reason": "negative_depth",
            "decode_error": None,
            "empty_stream": False,
            "in_train": None,
        },
        {
            "index": 5,
            "tokens": ["OPEN", "STOP"],
            "success": True,
            "invalid_reason": "decode_error",
            "decode_error": "unknown token 'OPEN' at position 0",
            "empty_stream": False,
            "in_train": None,
        },
    ]
