"""Split a one-Block Plan into one Plan per training job (its rescore jobs follow the source run).

Used to run the final seeds concurrently on separate Colab sessions:
    uv run python experiments/pretrain/split_plan.py --plan experiments/pretrain/plan_final.json --out-dir experiments/pretrain
writes plan_final_<job name>.json for every TrainJob of the single Block.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.runner.types import Block, Plan, plan_from_json, plan_to_json, validate_plan


def split(plan: Plan) -> list[Plan]:
    if len(plan.blocks) != 1:
        raise ValueError("split_plan expects a single Block")
    block = plan.blocks[0]
    parts = []
    for job in block.train:
        rescore = tuple(r for r in block.rescore if r.source_run == job.name)
        part = Plan(f"{plan.plan_id}_{job.name}", (Block(block.source_bundle, (job,), rescore),))
        validate_plan(part)
        parts.append(part)
    return parts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = plan_from_json(args.plan.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for part in split(plan):
        path = args.out_dir / f"{args.plan.stem}_{part.blocks[0].train[0].name}.json"
        path.write_text(plan_to_json(part), encoding="utf-8")
        print(path, "train", len(part.blocks[0].train), "rescore", len(part.blocks[0].rescore))


if __name__ == "__main__":
    main()
