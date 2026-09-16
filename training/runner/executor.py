"""Execute a Plan on a Backend, reproducing the shell runner step by step.

The executor stages the code and one bundle at a time, trains every seed,
fetches the six output files, runs the local evaluation, then re-scores the
trained checkpoints on the target bundles. It is idempotent (a run is done iff
``runs/<name>/test_scores.jsonl`` exists), removes partial output on failure,
re-acquires the backend after a lost session, records the backend used per run,
and can dry-run without touching anything. Standard library only: no torch,
no cfg_reducer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .backend import Backend, BackendUnavailable, RemoteFailure, SessionLost
from .fake import DryRunBackend
from .router import Router
from .scripts import rescore_jobs, rescore_script, train_wrapper
from .types import Block, Plan, bundle_name

TRAIN_OUTPUTS = (
    "samples.json",
    "history.json",
    "model.pt",
    "val_scores.jsonl",
    "test_scores.jsonl",
    "samples_constrained.json",
)
BUNDLE_FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "vocab.json", "meta.json")
CODE_FILES = ("training/train_ar.py", "training/grammar_mask.py")


class PlanExecutor:
    """Run ``plan`` on whatever backend the router hands over."""

    def __init__(
        self,
        plan: Plan,
        router: Router,
        *,
        repo_root: Path,
        runs_dir: Path,
        local_eval: Callable[[list[str]], int] | None = None,
        log: Callable[[str], None] = print,
        dry_run: bool = False,
        now: Callable[[], datetime] = datetime.now,
        train_timeout_s: int = 6000,
        rescore_timeout_s: int = 2400,
    ) -> None:
        self.plan = plan
        self.router = router
        self.repo_root = Path(repo_root)
        self.runs_dir = Path(runs_dir)
        self.local_eval: Callable[[list[str]], int] = (
            self._subprocess_eval if local_eval is None else local_eval
        )
        self.log = log
        self.dry_run = dry_run
        self.now = now
        self.train_timeout_s = train_timeout_s
        self.rescore_timeout_s = rescore_timeout_s
        self._dry_done: set[str] = set()
        self._started = ""

    def _subprocess_eval(self, argv: list[str]) -> int:
        completed = subprocess.run(list(argv), cwd=self.repo_root)
        return completed.returncode

    def stamp(self) -> str:
        return self.now().strftime("%H:%M:%S")

    def _iso(self) -> str:
        return self.now().isoformat(timespec="seconds")

    def done(self, name: str) -> bool:
        if (self.runs_dir / name / "test_scores.jsonl").exists():
            return True
        return self.dry_run and name in self._dry_done

    def _block_complete(self, block: Block) -> bool:
        return all(self.done(job.name) for job in block.train) and all(
            self.done(job.out_run) for job in block.rescore
        )

    def stage_code(self, backend: Backend) -> None:
        for relative in CODE_FILES:
            backend.put(
                self.repo_root / relative,
                f"{backend.root}/{Path(relative).name}",
            )

    def run(self) -> int:
        self._started = self._iso()
        if self.dry_run:
            backend: Backend = DryRunBackend(root="/content", log=self.log)
        else:
            try:
                backend = self.router.acquire()
            except BackendUnavailable:
                return 2
        self.stage_code(backend)

        for block in self.plan.blocks:
            if self._block_complete(block):
                self.log(f"skip {block.source_bundle} (complete)")
                continue
            if not backend.alive():
                self.log(f"SESSION-LOST before {block.source_bundle}")
                try:
                    backend = self.router.reacquire(backend)
                except BackendUnavailable:
                    return 2
                self.stage_code(backend)
            status, backend = self._attempt_block(backend, block)
            if status == 2:
                return 2
            if status == 3:
                backend.release()
                return 3

        backend.release()
        self.log(f"RUN-DONE {self.stamp()}")
        return 0

    def _attempt_block(self, backend: Backend, block: Block) -> tuple[int, Backend]:
        for attempt in (0, 1):
            try:
                self.run_block(backend, block)
                return 0, backend
            except SessionLost:
                self.log(f"SESSION-LOST at {block.source_bundle}")
                lost = True
            except RemoteFailure as exc:
                self.log(f"FAILURE at {block.source_bundle}: {exc}")
                lost = False
            if attempt == 1:
                return 3, backend
            if lost or not backend.alive():
                try:
                    backend = self.router.reacquire(backend)
                except BackendUnavailable:
                    return 2, backend
                self.stage_code(backend)
        return 3, backend

    def _eval_argv(self, block: Block, run_dir: Path) -> list[str]:
        return [
            "uv", "run", "python", "-m", "training.eval_samples",
            "--samples", str(run_dir / "samples.json"),
            "--vocab", f"{block.source_bundle}/vocab.json",
            "--train-tokens", f"{block.source_bundle}/train.jsonl",
            "--out", str(run_dir / "eval.json"),
        ]

    def _write_backend_json(self, backend: Backend, dest_dir: Path) -> None:
        data: dict[str, object] = {
            "finished": self._iso(),
            "kind": backend.kind,
            "plan_id": self.plan.plan_id,
            "started": self._started,
        }
        data.update(backend.describe())
        (dest_dir / "backend.json").write_text(
            json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def run_block(self, backend: Backend, block: Block) -> None:
        root = backend.root
        source_name = bundle_name(block.source_bundle)
        for name in BUNDLE_FILES:
            backend.put(
                self.repo_root / block.source_bundle / name,
                f"{root}/{name}",
            )
        backend.put(
            self.repo_root / block.source_bundle / "vocab.json",
            f"{root}/vocab_{source_name}.json",
        )
        backend.put(
            self.repo_root / block.source_bundle / "meta.json",
            f"{root}/meta_{source_name}.json",
        )

        for job in block.train:
            if self.done(job.name):
                self.log(f"skip {job.name}")
                continue
            self.log(f"=== {job.name} start {self.stamp()} ===")
            output = backend.run(
                train_wrapper(job, root), job.name, self.train_timeout_s
            )
            if output:
                self.log(output)
            run_dir = self.runs_dir / job.name
            if self.dry_run:
                for name in TRAIN_OUTPUTS:
                    backend.get(f"{root}/{job.name}/{name}", run_dir / name)
                self.log("EVAL " + " ".join(self._eval_argv(block, run_dir)))
                self._dry_done.add(job.name)
                self.log(f"=== {job.name} done {self.stamp()} ===")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            for name in TRAIN_OUTPUTS:
                backend.get(f"{root}/{job.name}/{name}", run_dir / name)
            if (run_dir / "test_scores.jsonl").exists():
                self.local_eval(self._eval_argv(block, run_dir))
                self._write_backend_json(backend, run_dir)
                self.log(f"=== {job.name} done {self.stamp()} ===")
            else:
                self.log(f"=== {job.name} FAILED {self.stamp()} ===")
                shutil.rmtree(run_dir, ignore_errors=True)
                raise RemoteFailure(f"{job.name}: no test_scores.jsonl")

        pending = [job for job in block.rescore if not self.done(job.out_run)]
        if not pending:
            return

        for target in sorted({job.target_bundle for job in pending}):
            backend.put(
                self.repo_root / target / "test.jsonl",
                f"{root}/test_{bundle_name(target)}.jsonl",
            )
        for source in sorted({job.source_run for job in pending}):
            backend.put(
                self.runs_dir / source / "model.pt",
                f"{root}/{source}_model.pt",
            )
            backend.put(
                self.runs_dir / source / "samples.json",
                f"{root}/{source}_samples.json",
            )

        jobs = rescore_jobs(
            Block(block.source_bundle, block.train, tuple(pending)), root
        )
        tmp_dir = Path(tempfile.mkdtemp(prefix="gr-runner-"))
        jobs_path = tmp_dir / "rescore_jobs.json"
        jobs_path.write_text(
            json.dumps(jobs, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        backend.put(jobs_path, f"{root}/rescore_jobs.json")

        self.log(f"=== rescore {source_name} {self.stamp()} ===")
        output = backend.run(
            rescore_script(root), f"rescore_{source_name}", self.rescore_timeout_s
        )
        if output:
            self.log(output)

        missing: list[str] = []
        for job in pending:
            target_name = bundle_name(job.target_bundle)
            out_dir = self.runs_dir / job.out_run
            remote = f"{root}/{job.source_run}__{target_name}.jsonl"
            if self.dry_run:
                backend.get(remote, out_dir / "test_scores.jsonl")
                self._dry_done.add(job.out_run)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            if backend.get(remote, out_dir / "test_scores.jsonl"):
                source_eval = self.runs_dir / job.source_run / "eval.json"
                if source_eval.exists():
                    shutil.copyfile(source_eval, out_dir / "eval.json")
                self._write_backend_json(backend, out_dir)
            else:
                try:
                    out_dir.rmdir()
                except OSError:
                    pass
                missing.append(job.out_run)
        if missing:
            raise RemoteFailure(
                "missing rescore output: " + ", ".join(sorted(missing))
            )
