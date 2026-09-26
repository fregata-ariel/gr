"""Plan types and JSON (de)serialisation for the compute backend router.

Standard library only: no torch, no cfg_reducer, no other training imports.
A ``Plan`` is the execution-agnostic description of an experiment; where it
runs is decided by a Backend chosen at start-up.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class TrainJob:
    name: str
    epochs: int
    patience: int
    num_samples: int
    constrained_samples: int
    seed: int
    sample_seed: int
    extra: tuple[str, ...] = ()

    def argv(self, root: str) -> tuple[str, ...]:
        return (
            "--out", f"{root}/{self.name}",
            "--epochs", str(self.epochs),
            "--patience", str(self.patience),
            "--num-samples", str(self.num_samples),
            "--test", f"{root}/test.jsonl",
            "--constrained-samples", str(self.constrained_samples),
            "--seed", str(self.seed),
            "--sample-seed", str(self.sample_seed),
            *self.extra,
        )


@dataclass(frozen=True)
class RescoreJob:
    source_run: str
    target_bundle: str
    out_run: str


@dataclass(frozen=True)
class Block:
    source_bundle: str
    train: tuple[TrainJob, ...]
    rescore: tuple[RescoreJob, ...] = ()


@dataclass(frozen=True)
class Plan:
    plan_id: str
    blocks: tuple[Block, ...]


def bundle_name(path: str) -> str:
    """Flat remote name of a bundle path (``data/tok_s32_lay`` -> ``tok_s32_lay``)."""
    return PurePosixPath(path).name


_TRAIN_KEYS = frozenset({
    "name", "epochs", "patience", "num_samples", "constrained_samples",
    "seed", "sample_seed", "extra",
})
_RESCORE_KEYS = frozenset({"source_run", "target_bundle", "out_run"})
_BLOCK_KEYS = frozenset({"source_bundle", "train", "rescore"})
_PLAN_KEYS = frozenset({"plan_id", "blocks"})


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _check_keys(data: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} keys: {unknown}")


def _sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where} must be a JSON list")
    return list(value)


def _train_job(data: Any) -> TrainJob:
    obj = _object(data, "TrainJob")
    _check_keys(obj, _TRAIN_KEYS, "TrainJob")
    return TrainJob(
        name=obj["name"],
        epochs=obj["epochs"],
        patience=obj["patience"],
        num_samples=obj["num_samples"],
        constrained_samples=obj["constrained_samples"],
        seed=obj["seed"],
        sample_seed=obj["sample_seed"],
        extra=tuple(_sequence(obj.get("extra", ()), "TrainJob.extra")),
    )


def _rescore_job(data: Any) -> RescoreJob:
    obj = _object(data, "RescoreJob")
    _check_keys(obj, _RESCORE_KEYS, "RescoreJob")
    return RescoreJob(
        source_run=obj["source_run"],
        target_bundle=obj["target_bundle"],
        out_run=obj["out_run"],
    )


def _block(data: Any) -> Block:
    obj = _object(data, "Block")
    _check_keys(obj, _BLOCK_KEYS, "Block")
    return Block(
        source_bundle=obj["source_bundle"],
        train=tuple(_train_job(job) for job in _sequence(obj["train"], "Block.train")),
        rescore=tuple(
            _rescore_job(job)
            for job in _sequence(obj.get("rescore", ()), "Block.rescore")
        ),
    )


def plan_to_json(plan: Plan) -> str:
    """Serialise with sorted keys, two-space indent, one trailing newline."""
    return json.dumps(asdict(plan), sort_keys=True, indent=2) + "\n"


def plan_from_json(text: str) -> Plan:
    """Rebuild a ``Plan``; JSON lists become tuples, unknown keys raise."""
    obj = _object(json.loads(text), "Plan")
    _check_keys(obj, _PLAN_KEYS, "Plan")
    return Plan(
        plan_id=obj["plan_id"],
        blocks=tuple(_block(block) for block in _sequence(obj["blocks"], "Plan.blocks")),
    )


def validate_plan(plan: Plan) -> None:
    """Raise ``ValueError`` on duplicate run names, dangling sources, clashes."""
    train_names = [job.name for block in plan.blocks for job in block.train]
    if len(set(train_names)) != len(train_names):
        raise ValueError("duplicate TrainJob names in plan")
    train_set = set(train_names)

    out_names = [job.out_run for block in plan.blocks for job in block.rescore]
    if len(set(out_names)) != len(out_names):
        raise ValueError("duplicate RescoreJob out_run names in plan")

    for block in plan.blocks:
        block_names = {job.name for job in block.train}
        for job in block.rescore:
            if job.source_run not in block_names:
                raise ValueError(
                    f"RescoreJob source_run {job.source_run!r} is not a "
                    f"TrainJob of block {block.source_bundle!r}"
                )
            if job.out_run in train_set:
                raise ValueError(
                    f"RescoreJob out_run {job.out_run!r} clashes with a "
                    f"TrainJob name"
                )
