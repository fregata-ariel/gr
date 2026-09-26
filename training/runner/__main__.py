"""Command line entry point for the compute backend router.

``run`` executes a Plan JSON on the backend chosen by policy (command line,
else the ``GR_BACKEND`` environment variable, else ``colab``); ``--dry-run``
prints the whole command sequence without touching any backend. ``sweep``
writes the milestone sweep Plan as JSON. Standard library only: no torch, no
cfg_reducer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .colab import ColabBackend, ColabTimeouts
from .docker import DockerBackend
from .executor import PlanExecutor
from .plans import sweep_plan
from .router import Router, resolve_policy
from .types import plan_from_json, plan_to_json, validate_plan

DEFAULT_IMAGE = "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
DEFAULT_SESSION = "sw"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZES = "12,16,24,32,48"
DEFAULT_SOURCES = "lay,mix"


def _log(line: str) -> None:
    print(line, flush=True)


def _csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.runner",
        description="Plan and run the milestone sweep on a compute backend.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute a Plan JSON")
    run.add_argument("--plan", required=True, help="Plan JSON file")
    run.add_argument(
        "--backend", choices=("colab", "local", "auto"), default=None,
        help="execution target (default: GR_BACKEND, else colab)",
    )
    run.add_argument("--session", default=DEFAULT_SESSION, help="Colab session name")
    run.add_argument(
        "--gpu",
        default=os.environ.get("GR_COLAB_GPU", "T4"),
        help="Colab GPU variant (T4, L4, A100, ...); default GR_COLAB_GPU or T4",
    )
    run.add_argument(
        "--image",
        default=os.environ.get("GR_DOCKER_IMAGE", DEFAULT_IMAGE),
        help="Docker image for the local backend",
    )
    run.add_argument(
        "--staging", default=None,
        help="local staging directory (default: <repo-root>/.runner_staging)",
    )
    run.add_argument("--runs-dir", default="runs", help="output runs directory")
    run.add_argument(
        "--repo-root", default=str(DEFAULT_REPO_ROOT),
        help="repository root holding training/ and data/",
    )
    run.add_argument("--attempts", type=int, default=36, help="Colab allocation tries")
    run.add_argument("--sleep-s", type=int, default=600, help="seconds between tries")
    run.add_argument("--train-timeout-s", type=int, default=6000,
                     help="学習実行の outer timeout (秒、既定6000)")
    run.add_argument("--colab-inner-timeout-s", type=int, default=5400,
                     help="Colab exec の inner timeout (秒、既定5400)")
    run.add_argument("--rescore-timeout-s", type=int, default=2400)
    run.add_argument("--dry-run", action="store_true", help="print, touch nothing")
    run.add_argument(
        "--allow-backend-switch", action="store_true",
        help="allow switching target kind when re-acquiring",
    )

    sweep = subparsers.add_parser("sweep", help="write the milestone sweep Plan")
    sweep.add_argument("--sizes", default=DEFAULT_SIZES, help="comma separated sizes")
    sweep.add_argument("--sources", default=DEFAULT_SOURCES, help="comma separated sources")
    sweep.add_argument("--out", required=True, help="Plan JSON output file")
    return parser


def _run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    staging = (
        Path(args.staging)
        if args.staging is not None
        else repo_root / ".runner_staging"
    )
    try:
        policy = resolve_policy(args.backend)
        plan = plan_from_json(Path(args.plan).read_text(encoding="utf-8"))
        validate_plan(plan)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    factories = {
        "colab": lambda: ColabBackend(
            args.session, args.gpu, attempts=args.attempts, sleep_s=args.sleep_s, log=_log,
            timeouts=ColabTimeouts(exec_inner=args.colab_inner_timeout_s)
        ),
        "local": lambda: DockerBackend(args.image, staging, log=_log),
    }
    router = Router(
        factories, policy, allow_switch=args.allow_backend_switch, log=_log
    )
    executor = PlanExecutor(
        plan,
        router,
        repo_root=repo_root,
        runs_dir=Path(args.runs_dir),
        dry_run=args.dry_run,
        train_timeout_s=args.train_timeout_s,
        rescore_timeout_s=args.rescore_timeout_s,
        log=_log,
    )
    try:
        return executor.run()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4


def _sweep(args: argparse.Namespace) -> int:
    try:
        sizes = tuple(int(part) for part in _csv(args.sizes))
        sources = tuple(_csv(args.sources))
        plan = sweep_plan(sizes, sources)
        Path(args.out).write_text(plan_to_json(plan), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    print(f"plan_id={plan.plan_id} blocks={len(plan.blocks)}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 4
    if args.command == "sweep":
        return _sweep(args)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
