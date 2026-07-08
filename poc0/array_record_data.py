from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np

from poc0.constants import MAX_SEQ_LEN, SPLIT_SEED, VOCAB_SIZE
from poc0.data import Batch, DatasetInfo, _Record, _load_jsonl_records, _split_records
from poc0.tokenizer import TokenExample, record_to_example


def _require_linux(operation: str) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            f"{operation} requires Linux because grain/array_record wheels are only supported on Linux"
        )


def _require_grain(operation: str) -> Any:
    _require_linux(operation)
    try:
        return importlib.import_module("grain")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{operation} requires grain on Linux for poc0 ArrayRecord datasets"
        ) from exc


def _load_array_record_module(operation: str) -> Any:
    _require_linux(operation)
    candidates = (
        "array_record",
        "array_record.python.array_record_module",
    )
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    raise RuntimeError(f"{operation} requires array_record on Linux for poc0 conversion")


def _arrayrecord_writer(path: Path) -> Any:
    module = _load_array_record_module("convert_jsonl_to_arrayrecord")
    writer_class = getattr(module, "ArrayRecordWriter", None)
    if writer_class is None:
        raise RuntimeError("array_record does not expose ArrayRecordWriter")

    constructor_attempts = (
        lambda: writer_class(str(path)),
        lambda: writer_class(str(path), "group_size:1"),
        lambda: writer_class(str(path), options="group_size:1"),
    )
    last_error: Exception | None = None
    for attempt in constructor_attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc

    assert last_error is not None
    raise RuntimeError("Unable to construct ArrayRecordWriter") from last_error


def _arrayrecord_reader(path: Path) -> Any | None:
    module = _load_array_record_module("GrainArrayRecordDataset")
    reader_class = getattr(module, "ArrayRecordReader", None)
    if reader_class is None:
        return None

    constructor_attempts = (
        lambda: reader_class(str(path)),
        lambda: reader_class(str(path), ""),
    )
    for attempt in constructor_attempts:
        try:
            return attempt()
        except TypeError:
            continue

    return None


def _iter_reader_records(reader: Any) -> Iterator[bytes]:
    try:
        if hasattr(reader, "__iter__"):
            for record in reader:
                yield bytes(record)
            return

        if hasattr(reader, "__len__") and hasattr(reader, "__getitem__"):
            for index in range(len(reader)):
                yield bytes(reader[index])
            return

        if hasattr(reader, "read"):
            while True:
                try:
                    record = reader.read()
                except StopIteration:
                    return
                if record is None:
                    return
                yield bytes(record)
            return

        raise RuntimeError("Unsupported ArrayRecordReader interface")
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def _iter_arrayrecord_bytes(path: Path) -> Iterator[bytes]:
    direct_reader = _arrayrecord_reader(path)
    if direct_reader is not None:
        yield from _iter_reader_records(direct_reader)
        return

    grain_module = _require_grain("GrainArrayRecordDataset")
    grain_python = getattr(grain_module, "python", None)
    data_source_class = getattr(grain_module, "ArrayRecordDataSource", None)
    if data_source_class is None and grain_python is not None:
        data_source_class = getattr(grain_python, "ArrayRecordDataSource", None)
    if data_source_class is None:
        raise RuntimeError(
            "Unable to read ArrayRecord: neither ArrayRecordReader nor Grain ArrayRecordDataSource is available"
        )

    data_source = data_source_class(str(path))
    for index in range(len(data_source)):
        yield bytes(data_source[index])


def _serialize_example(record_id: str, example: TokenExample) -> bytes:
    payload = {
        "record_id": record_id,
        "input_ids": example.input_ids.tolist(),
        "target_ids": example.target_ids.tolist(),
        "loss_mask": example.loss_mask.astype(bool, copy=False).tolist(),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _validate_int_list(field_name: str, values: list[object]) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"ArrayRecord {field_name}[{index}] must be an int, got {type(value).__name__}"
            )


def _validate_bool_list(field_name: str, values: list[object]) -> None:
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            raise ValueError(
                f"ArrayRecord {field_name}[{index}] must be a bool, got {type(value).__name__}"
            )


def _deserialize_example(payload: bytes) -> tuple[str, TokenExample]:
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError("ArrayRecord item must decode to an object")

    record_id = record.get("record_id")
    input_ids = record.get("input_ids")
    target_ids = record.get("target_ids")
    loss_mask = record.get("loss_mask")

    if not isinstance(record_id, str) or not record_id:
        raise ValueError("ArrayRecord item is missing record_id")
    if not isinstance(input_ids, list) or not isinstance(target_ids, list):
        raise ValueError("ArrayRecord item is missing token id arrays")
    if not isinstance(loss_mask, list):
        raise ValueError("ArrayRecord item is missing loss_mask")

    _validate_int_list("input_ids", input_ids)
    _validate_int_list("target_ids", target_ids)
    _validate_bool_list("loss_mask", loss_mask)

    input_array = np.asarray(input_ids, dtype=np.int32)
    target_array = np.asarray(target_ids, dtype=np.int32)
    loss_mask_array = np.asarray(loss_mask, dtype=np.bool_)

    expected_shape = (MAX_SEQ_LEN,)
    if input_array.shape != expected_shape:
        raise ValueError(
            f"ArrayRecord input_ids must have shape {expected_shape}, got {input_array.shape}"
        )
    if target_array.shape != expected_shape:
        raise ValueError(
            f"ArrayRecord target_ids must have shape {expected_shape}, got {target_array.shape}"
        )
    if loss_mask_array.shape != expected_shape:
        raise ValueError(
            f"ArrayRecord loss_mask must have shape {expected_shape}, got {loss_mask_array.shape}"
        )

    return record_id, TokenExample(
        input_ids=input_array,
        target_ids=target_array,
        loss_mask=loss_mask_array,
    )


def _write_array_record_file(path: Path, records: Sequence[_Record]) -> None:
    serialized_examples = [
        _serialize_example(record.record_id, record_to_example(record.tokens))
        for record in records
    ]
    writer = _arrayrecord_writer(path)
    try:
        for item in serialized_examples:
            writer.write(item)
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()


def _build_manifest(
    *,
    jsonl_path: Path,
    out_dir: Path,
    train_records: Sequence[_Record],
    val_records: Sequence[_Record],
    max_seq_len: int,
    split_seed: int,
) -> dict[str, object]:
    del out_dir
    return {
        "version": 1,
        "source_path": str(jsonl_path),
        "max_seq_len": max_seq_len,
        "vocab_size": VOCAB_SIZE,
        "split_seed": split_seed,
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


def convert_jsonl_to_arrayrecord(
    *,
    jsonl_path: Path,
    out_dir: Path,
    max_seq_len: int = MAX_SEQ_LEN,
    split_seed: int = SPLIT_SEED,
) -> Path:
    _require_linux("convert_jsonl_to_arrayrecord")
    _require_grain("convert_jsonl_to_arrayrecord")
    _load_array_record_module("convert_jsonl_to_arrayrecord")

    if max_seq_len != MAX_SEQ_LEN:
        raise ValueError(
            f"max_seq_len must be {MAX_SEQ_LEN} to match the Stage 1 tokenizer, got {max_seq_len}"
        )

    jsonl_path = Path(jsonl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = _load_jsonl_records(jsonl_path)
    train_records = _split_records(
        all_records,
        split="train",
        split_seed=split_seed,
    )
    val_records = _split_records(
        all_records,
        split="val",
        split_seed=split_seed,
    )

    _write_array_record_file(out_dir / "train.array_record", train_records)
    _write_array_record_file(out_dir / "val.array_record", val_records)

    manifest = _build_manifest(
        jsonl_path=jsonl_path,
        out_dir=out_dir,
        train_records=train_records,
        val_records=val_records,
        max_seq_len=max_seq_len,
        split_seed=split_seed,
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _stack_examples_int32(examples: Sequence[TokenExample], field: str) -> np.ndarray:
    if not examples:
        return np.empty((0, MAX_SEQ_LEN), dtype=np.int32)
    arrays = [getattr(example, field) for example in examples]
    return np.stack(arrays, axis=0).astype(np.int32, copy=False)


def _stack_examples_bool(examples: Sequence[TokenExample], field: str) -> np.ndarray:
    if not examples:
        return np.empty((0, MAX_SEQ_LEN), dtype=np.bool_)
    arrays = [getattr(example, field) for example in examples]
    return np.stack(arrays, axis=0).astype(np.bool_, copy=False)


class GrainArrayRecordDataset:
    def __init__(
        self,
        *,
        manifest_path: Path,
        split: Literal["train", "val"],
    ) -> None:
        _require_linux("GrainArrayRecordDataset")
        _require_grain("GrainArrayRecordDataset")
        _load_array_record_module("GrainArrayRecordDataset")

        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self._manifest_path = Path(manifest_path)
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must contain an object")

        splits = manifest.get("splits")
        if not isinstance(splits, dict):
            raise ValueError("manifest.json is missing splits")
        split_manifest = splits.get(split)
        if not isinstance(split_manifest, dict):
            raise ValueError(f"manifest.json is missing split {split}")

        relative_path = split_manifest.get("path")
        record_ids = split_manifest.get("record_ids")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"manifest split {split} is missing a path")
        if not isinstance(record_ids, list) or not all(
            isinstance(record_id, str) for record_id in record_ids
        ):
            raise ValueError(f"manifest split {split} is missing record_ids")

        array_record_path = self._manifest_path.parent / relative_path

        loaded_record_ids: list[str] = []
        examples: list[TokenExample] = []
        for item in _iter_arrayrecord_bytes(array_record_path):
            record_id, example = _deserialize_example(item)
            loaded_record_ids.append(record_id)
            examples.append(example)

        if loaded_record_ids != record_ids:
            raise ValueError(
                f"ArrayRecord contents for split {split} do not match manifest record_ids"
            )

        manifest_max_seq_len = manifest.get("max_seq_len")
        manifest_vocab_size = manifest.get("vocab_size")
        if manifest_max_seq_len != MAX_SEQ_LEN:
            raise ValueError(
                f"manifest max_seq_len must be {MAX_SEQ_LEN}, got {manifest_max_seq_len}"
            )
        if manifest_vocab_size != VOCAB_SIZE:
            raise ValueError(
                f"manifest vocab_size must be {VOCAB_SIZE}, got {manifest_vocab_size}"
            )

        self._manifest = manifest
        self._record_ids = tuple(record_ids)
        self._input_ids = _stack_examples_int32(examples, "input_ids")
        self._target_ids = _stack_examples_int32(examples, "target_ids")
        self._loss_mask = _stack_examples_bool(examples, "loss_mask")
        self._info = DatasetInfo(
            split=split,
            n_records=len(examples),
            max_seq_len=MAX_SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            source_path=str(array_record_path),
            manifest=manifest,
        )

    @property
    def info(self) -> DatasetInfo:
        return self._info

    @property
    def manifest(self) -> dict[str, object]:
        return self._manifest

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m poc0.array_record_data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--jsonl", type=Path, required=True)
    convert_parser.add_argument("--out-dir", type=Path, required=True)
    convert_parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    convert_parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)

    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command != "convert":
        raise ValueError(f"Unsupported command: {args.command}")

    manifest_path = convert_jsonl_to_arrayrecord(
        jsonl_path=args.jsonl,
        out_dir=args.out_dir,
        max_seq_len=args.max_seq_len,
        split_seed=args.split_seed,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
