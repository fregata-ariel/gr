from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from poc0.constants import MAX_SEQ_LEN, TOKEN_TO_ID
from poc0.data import InMemoryTokenDataset


def _write_records(
    path: Path,
    n_records: int,
) -> list[dict[str, str | list[str]]]:
    records: list[dict[str, str | list[str]]] = []
    with path.open("w", encoding="utf-8") as fh:
        for index in range(n_records):
            record = {
                "id": f"record-{index:02d}",
                "tokens": ["ADD_ENTRY", f"ptr_{index}", "STOP"],
            }
            records.append(record)
            fh.write(json.dumps(record))
            fh.write("\n")
    return records


def test_split_is_seeded_and_stable(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    records = _write_records(jsonl_path, n_records=20)
    split_seed = 20260707

    train_a = InMemoryTokenDataset(
        jsonl_path=jsonl_path,
        split="train",
        split_seed=split_seed,
    )
    train_b = InMemoryTokenDataset(
        jsonl_path=jsonl_path,
        split="train",
        split_seed=split_seed,
    )
    val = InMemoryTokenDataset(
        jsonl_path=jsonl_path,
        split="val",
        split_seed=split_seed,
    )

    shuffled_ids = [record["id"] for record in records]
    random.Random(split_seed).shuffle(shuffled_ids)
    train_cutoff = int(0.95 * len(records))
    expected_train_ids = tuple(shuffled_ids[:train_cutoff])
    expected_val_ids = tuple(shuffled_ids[train_cutoff:])

    assert train_a.record_ids == expected_train_ids
    assert train_b.record_ids == expected_train_ids
    assert val.record_ids == expected_val_ids
    assert set(train_a.record_ids).isdisjoint(val.record_ids)


def test_in_memory_source_yields_batch_contract(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    _write_records(jsonl_path, n_records=20)
    train_dataset = InMemoryTokenDataset(
        jsonl_path=jsonl_path,
        split="train",
        split_seed=20260707,
    )

    first_iter = train_dataset.iter_batches(
        batch_size=4,
        shuffle=True,
        seed=17,
        drop_remainder=False,
        repeat=False,
    )
    second_iter = train_dataset.iter_batches(
        batch_size=4,
        shuffle=True,
        seed=17,
        drop_remainder=False,
        repeat=False,
    )

    first_batches = list(first_iter)
    second_batches = list(second_iter)

    assert len(first_batches) == 5
    assert len(second_batches) == 5
    assert len(
        list(
            train_dataset.iter_batches(
                batch_size=4,
                shuffle=False,
                seed=0,
                drop_remainder=True,
                repeat=False,
            )
        )
    ) == 4

    for batch_a, batch_b in zip(first_batches, second_batches, strict=True):
        assert batch_a.input_ids.shape[1:] == (MAX_SEQ_LEN,)
        assert batch_a.target_ids.shape[1:] == (MAX_SEQ_LEN,)
        assert batch_a.loss_mask.shape[1:] == (MAX_SEQ_LEN,)
        assert batch_a.input_ids.dtype == np.int32
        assert batch_a.target_ids.dtype == np.int32
        assert batch_a.loss_mask.dtype == np.bool_
        assert np.array_equal(batch_a.input_ids, batch_b.input_ids)
        assert np.array_equal(batch_a.target_ids, batch_b.target_ids)
        assert np.array_equal(batch_a.loss_mask, batch_b.loss_mask)

    finite_ids = [
        int(row[2])
        for batch in train_dataset.iter_batches(
            batch_size=4,
            shuffle=False,
            seed=0,
            drop_remainder=False,
            repeat=False,
        )
        for row in batch.input_ids
    ]
    repeated_iter = train_dataset.iter_batches(
        batch_size=4,
        shuffle=False,
        seed=0,
        drop_remainder=False,
        repeat=True,
    )
    repeated_ids: list[int] = []
    while len(repeated_ids) < len(finite_ids) * 2:
        batch = next(repeated_iter)
        repeated_ids.extend(int(row[2]) for row in batch.input_ids)
    repeated_ids = repeated_ids[: len(finite_ids) * 2]

    assert repeated_ids[: len(finite_ids)] == finite_ids
    assert repeated_ids[len(finite_ids):] == finite_ids

    base_epoch_ids = [
        TOKEN_TO_ID[
            f"ptr_{int(record_id.removeprefix('record-'))}"
        ]
        for record_id in train_dataset.record_ids
    ]
    expected_epoch0_ids = list(base_epoch_ids)
    expected_epoch1_ids = list(base_epoch_ids)
    random.Random(17).shuffle(expected_epoch0_ids)
    random.Random(18).shuffle(expected_epoch1_ids)

    shuffled_repeat = train_dataset.iter_batches(
        batch_size=5,
        shuffle=True,
        seed=17,
        drop_remainder=False,
        repeat=True,
    )
    observed_ids: list[int] = []
    while len(observed_ids) < len(base_epoch_ids) * 2:
        batch = next(shuffled_repeat)
        observed_ids.extend(int(row[2]) for row in batch.input_ids)
    observed_ids = observed_ids[: len(base_epoch_ids) * 2]

    assert observed_ids[: len(base_epoch_ids)] == expected_epoch0_ids
    assert observed_ids[len(base_epoch_ids):] == expected_epoch1_ids
