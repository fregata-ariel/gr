from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import poc0.array_record_data as array_record_data
from poc0.constants import MAX_SEQ_LEN, SPLIT_SEED, VOCAB_SIZE
from poc0.data import InMemoryTokenDataset, _load_jsonl_records, _split_records


def _write_records(
    path: Path,
    n_records: int,
) -> list[dict[str, str | list[str]]]:
    records: list[dict[str, str | list[str]]] = []
    with path.open("w", encoding="utf-8") as fh:
        for index in range(n_records):
            record = {
                "id": f"record-{index:02d}",
                "tokens": [
                    "ADD_ENTRY",
                    f"ptr_{index}",
                    "ADD_LOOP",
                    f"ptr_{index + 1}",
                    "OPEN",
                    "ADD_LINEAR",
                    f"ptr_{index + 2}",
                    "STOP",
                    "CLOSE",
                    "STOP",
                ],
            }
            records.append(record)
            fh.write(json.dumps(record))
            fh.write("\n")
    return records


def test_build_manifest_tracks_split_ids_and_relative_paths(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    out_dir = tmp_path / "arrayrecord"
    _write_records(jsonl_path, n_records=20)

    all_records = _load_jsonl_records(jsonl_path)
    train_records = _split_records(all_records, split="train", split_seed=SPLIT_SEED)
    val_records = _split_records(all_records, split="val", split_seed=SPLIT_SEED)

    manifest = array_record_data._build_manifest(
        jsonl_path=jsonl_path,
        out_dir=out_dir,
        train_records=train_records,
        val_records=val_records,
        max_seq_len=MAX_SEQ_LEN,
        split_seed=SPLIT_SEED,
    )

    assert manifest == {
        "version": 1,
        "source_path": str(jsonl_path),
        "max_seq_len": MAX_SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "split_seed": SPLIT_SEED,
        "splits": {
            "train": {
                "split": "train",
                "n_records": len(train_records),
                "path": "train.array_record",
                "record_ids": [record.record_id for record in train_records],
            },
            "val": {
                "split": "val",
                "n_records": len(val_records),
                "path": "val.array_record",
                "record_ids": [record.record_id for record in val_records],
            },
        },
    }


def test_parse_args_for_convert_command():
    args = array_record_data._parse_args(
        [
            "convert",
            "--jsonl",
            "/tmp/poc0_corpus.jsonl",
            "--out-dir",
            "/tmp/arrayrecord",
            "--max-seq-len",
            "80",
            "--split-seed",
            "20260707",
        ]
    )

    assert args.command == "convert"
    assert args.jsonl == Path("/tmp/poc0_corpus.jsonl")
    assert args.out_dir == Path("/tmp/arrayrecord")
    assert args.max_seq_len == 80
    assert args.split_seed == 20260707


def test_require_linux_raises_clear_error_off_linux(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(array_record_data.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="Linux"):
        array_record_data._require_linux("convert_jsonl_to_arrayrecord")


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("input_ids", [1.9] * MAX_SEQ_LEN),
        ("input_ids", ["1"] * MAX_SEQ_LEN),
        ("input_ids", [True] * MAX_SEQ_LEN),
        ("target_ids", [1.9] * MAX_SEQ_LEN),
        ("target_ids", ["1"] * MAX_SEQ_LEN),
        ("target_ids", [False] * MAX_SEQ_LEN),
        ("loss_mask", [1] * MAX_SEQ_LEN),
        ("loss_mask", ["yes"] * MAX_SEQ_LEN),
    ],
)
def test_deserialize_example_rejects_invalid_element_types_before_coercion(
    field_name: str,
    field_value: list[object],
):
    payload = {
        "record_id": "record-01",
        "input_ids": [0] * MAX_SEQ_LEN,
        "target_ids": [0] * MAX_SEQ_LEN,
        "loss_mask": [False] * MAX_SEQ_LEN,
    }
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        array_record_data._deserialize_example(
            json.dumps(payload).encode("utf-8")
        )


def test_arrayrecord_smoke_matches_in_memory_source_or_skips(tmp_path: Path):
    if sys.platform != "linux":
        pytest.skip("requires Linux")

    pytest.importorskip("grain")
    pytest.importorskip("array_record")

    jsonl_path = tmp_path / "records.jsonl"
    out_dir = tmp_path / "arrayrecord"
    _write_records(jsonl_path, n_records=20)

    manifest_path = array_record_data.convert_jsonl_to_arrayrecord(
        jsonl_path=jsonl_path,
        out_dir=out_dir,
        max_seq_len=MAX_SEQ_LEN,
        split_seed=SPLIT_SEED,
    )

    for split in ("train", "val"):
        expected = InMemoryTokenDataset(
            jsonl_path=jsonl_path,
            split=split,
            split_seed=SPLIT_SEED,
        )
        actual = array_record_data.GrainArrayRecordDataset(
            manifest_path=manifest_path,
            split=split,
        )

        assert actual.info.split == expected.info.split
        assert actual.info.n_records == expected.info.n_records
        assert actual.info.max_seq_len == expected.info.max_seq_len
        assert actual.info.vocab_size == expected.info.vocab_size
        assert actual.info.manifest == actual.manifest

        expected_batches = list(
            expected.iter_batches(
                batch_size=4,
                shuffle=True,
                seed=17,
                drop_remainder=False,
                repeat=False,
            )
        )
        actual_batches = list(
            actual.iter_batches(
                batch_size=4,
                shuffle=True,
                seed=17,
                drop_remainder=False,
                repeat=False,
            )
        )

        assert len(actual_batches) == len(expected_batches)
        for expected_batch, actual_batch in zip(
            expected_batches,
            actual_batches,
            strict=True,
        ):
            assert actual_batch.input_ids.dtype == np.int32
            assert actual_batch.target_ids.dtype == np.int32
            assert actual_batch.loss_mask.dtype == np.bool_
            assert np.array_equal(actual_batch.input_ids, expected_batch.input_ids)
            assert np.array_equal(actual_batch.target_ids, expected_batch.target_ids)
            assert np.array_equal(actual_batch.loss_mask, expected_batch.loss_mask)
