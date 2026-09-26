"""Milestone sweep Plan builder (see docs/design/generator_v2.md).

Reproduces the 30 training wrappers and 75 rescore jobs of the n12-n48 sweep.
"""

from __future__ import annotations

from collections.abc import Iterable

from .types import Block, Plan, RescoreJob, TrainJob

SWEEP_SOURCES: tuple[str, ...] = ("lay", "mix")
SWEEP_TARGETS: dict[str, tuple[str, ...]] = {
    "lay": ("str", "spa"),
    "mix": ("str", "spa", "lay"),
}


def sweep_block(size: int, source: str) -> Block:
    """One (size, source) block: three mask runs re-scored on each target."""
    source_bundle = f"data/tok_s{size}_{source}"
    train = tuple(
        TrainJob(
            name=f"c_s{size}_{source}_mask_n{size}_s{seed}",
            epochs=300,
            patience=20,
            num_samples=400,
            constrained_samples=400,
            seed=seed,
            sample_seed=1000 + seed,
            extra=("--ref-legal-mask",),
        )
        for seed in range(3)
    )
    rescore = tuple(
        RescoreJob(
            source_run=job.name,
            target_bundle=f"data/tok_s{size}_{source}2{short}",
            out_run=f"c_s{size}_{source}2{short}_mask_n{size}_s{seed}",
        )
        for seed, job in enumerate(train)
        for short in SWEEP_TARGETS[source]
    )
    return Block(source_bundle=source_bundle, train=train, rescore=rescore)


def sweep_plan(
    sizes: Iterable[int], sources: Iterable[str] = SWEEP_SOURCES
) -> Plan:
    """Build the milestone sweep: one block per (size, source)."""
    size_list = tuple(int(size) for size in sizes)
    blocks = tuple(
        sweep_block(size, source)
        for size in size_list
        for source in sources
    )
    plan_id = "sweep_" + "_".join(str(size) for size in size_list)
    return Plan(plan_id=plan_id, blocks=blocks)
