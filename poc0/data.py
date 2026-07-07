from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol

import numpy as np

from poc0.constants import MAX_SEQ_LEN, SPLIT_SEED, TOKEN_TO_ID, TRAIN_SPLIT_FRACTION, VOCAB_SIZE
from poc0.tokenizer import record_to_example


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray


@dataclass(frozen=True)
class DatasetInfo:
    split: str
    n_records: int
    max_seq_len: int
    vocab_size: int
    source_path: str
    manifest: dict[str, object]


class BatchSource(Protocol):
    @property
    def info(self) -> DatasetInfo: ...

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool,
        repeat: bool,
    ) -> Iterator[Batch]: ...


@dataclass(frozen=True)
class _Record:
    record_id: str
    tokens: list[str]


def _load_jsonl_records(jsonl_path: Path) -> list[_Record]:
    records: list[_Record] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL line {line_number} must contain an object record"
                )

            record_id = payload.get("id")
            tokens = payload.get("tokens")
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

            records.append(_Record(record_id=record_id, tokens=tokens))

    return sorted(records, key=lambda record: record.record_id)


def _split_records(
    records: list[_Record],
    split: Literal["train", "val"],
    split_seed: int,
) -> list[_Record]:
    shuffled_records = list(records)
    random.Random(split_seed).shuffle(shuffled_records)
    train_size = int(TRAIN_SPLIT_FRACTION * len(shuffled_records))
    if split == "train":
        return shuffled_records[:train_size]
    return shuffled_records[train_size:]


class InMemoryTokenDataset:
    def __init__(
        self,
        *,
        jsonl_path: Path,
        split: Literal["train", "val"],
        split_seed: int = SPLIT_SEED,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self._jsonl_path = Path(jsonl_path)
        all_records = _load_jsonl_records(self._jsonl_path)
        split_records = _split_records(all_records, split=split, split_seed=split_seed)

        self._record_ids = tuple(record.record_id for record in split_records)
        examples = [record_to_example(record.tokens) for record in split_records]

        self._input_ids = np.stack(
            [example.input_ids for example in examples],
            axis=0,
        ).astype(np.int32, copy=False)
        self._target_ids = np.stack(
            [example.target_ids for example in examples],
            axis=0,
        ).astype(np.int32, copy=False)
        self._loss_mask = np.stack(
            [example.loss_mask for example in examples],
            axis=0,
        ).astype(np.bool_, copy=False)

        self._info = DatasetInfo(
            split=split,
            n_records=len(split_records),
            max_seq_len=MAX_SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            source_path=str(self._jsonl_path),
            manifest={
                "record_ids": self._record_ids,
                "split_seed": split_seed,
                "vocab_size": VOCAB_SIZE,
                "max_seq_len": MAX_SEQ_LEN,
            },
        )

    @property
    def info(self) -> DatasetInfo:
        return self._info

    @property
    def record_ids(self) -> tuple[str, ...]:
        return self._record_ids

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool,
        repeat: bool,
    ) -> Iterator[Batch]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        n_records = self._input_ids.shape[0]
        epoch_index = 0

        while True:
            indices = list(range(n_records))
            if shuffle:
                random.Random(seed + epoch_index).shuffle(indices)

            for start in range(0, n_records, batch_size):
                stop = min(start + batch_size, n_records)
                if drop_remainder and stop - start < batch_size:
                    break

                batch_indices = indices[start:stop]
                yield Batch(
                    input_ids=self._input_ids[batch_indices],
                    target_ids=self._target_ids[batch_indices],
                    loss_mask=self._loss_mask[batch_indices],
                )

            if not repeat:
                return
            epoch_index += 1
