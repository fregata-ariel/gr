"""Tests for PlanExecutor against FakeBackend and the DryRunBackend.

Nothing leaves the process: training runs are fabricated in the fake backend's
in-memory file store, local_eval is a recorder, and every path lives under
``tmp_path``. The real router, Colab and Docker backends are never used.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from training.runner import (
    BUNDLE_FILES,
    TRAIN_OUTPUTS,
    Block,
    FakeBackend,
    Plan,
    PlanExecutor,
    RescoreJob,
    Router,
    SessionLost,
    TrainJob,
    train_wrapper,
)

NOW = datetime(2026, 9, 16, 12, 0, 0)
ISO_NOW = "2026-09-16T12:00:00"


def make_plan() -> Plan:
    block = Block(
        source_bundle="data/tok_x",
        train=(
            TrainJob("r_s0", 2, 1, 5, 5, 0, 1000, ("--ref-legal-mask",)),
            TrainJob("r_s1", 2, 1, 5, 5, 1, 1001, ("--ref-legal-mask",)),
        ),
        rescore=(
            RescoreJob("r_s0", "data/tok_x2y", "r2y_s0"),
            RescoreJob("r_s1", "data/tok_x2y", "r2y_s1"),
        ),
    )
    return Plan("p1", (block,))


PLAN = make_plan()


class RecordingEval:
    """Records argv and writes the ``--out`` file, mimicking eval_samples."""

    def __init__(self) -> None:
        self.argvs: list[list[str]] = []

    def __call__(self, argv: list[str]) -> int:
        self.argvs.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}\n", encoding="utf-8")
        return 0


def build_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    training = repo_root / "training"
    training.mkdir(parents=True)
    (training / "train_ar.py").write_text("", encoding="utf-8")
    (training / "grammar_mask.py").write_text("", encoding="utf-8")
    bundle = repo_root / "data" / "tok_x"
    bundle.mkdir(parents=True)
    for name in BUNDLE_FILES:
        (bundle / name).write_bytes(b"bundle\n")
    target = repo_root / "data" / "tok_x2y"
    target.mkdir(parents=True)
    (target / "test.jsonl").write_bytes(b"target\n")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return repo_root, runs_dir


def make_backend(
    *,
    omit_r_s1_first: bool = False,
    omit_r_s1_always: bool = False,
    omit_rescore: bool = False,
) -> FakeBackend:
    backend = FakeBackend()
    backend.kind = "colab"
    train_names = {job.name for job in PLAN.blocks[0].train}
    counts = {"r_s1": 0}

    def on_run(name: str, script: str) -> str:
        if name in train_names:
            if name == "r_s1":
                counts["r_s1"] += 1
            omit = (name == "r_s1") and (
                omit_r_s1_always
                or (omit_r_s1_first and counts["r_s1"] == 1)
            )
            for output in TRAIN_OUTPUTS:
                if omit and output == "test_scores.jsonl":
                    continue
                backend.files[f"/fake/{name}/{output}"] = b"out\n"
        elif name.startswith("rescore_"):
            if omit_rescore:
                return ""
            jobs = json.loads(
                backend.files["/fake/rescore_jobs.json"].decode("utf-8")
            )
            for job in jobs:
                backend.files[job["out"]] = b"scores\n"
        return ""

    backend.on_run = on_run
    return backend


def make_executor(
    repo_root: Path,
    runs_dir: Path,
    backend: FakeBackend,
    logs: list[str],
    *,
    dry_run: bool = False,
) -> tuple[PlanExecutor, RecordingEval]:
    evalr = RecordingEval()
    router = Router({"colab": lambda: backend}, "colab", log=logs.append)
    executor = PlanExecutor(
        PLAN,
        router,
        repo_root=repo_root,
        runs_dir=runs_dir,
        local_eval=evalr,
        log=logs.append,
        dry_run=dry_run,
        now=lambda: NOW,
    )
    return executor, evalr


# --- happy path --------------------------------------------------------------

def test_happy_path(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    backend = make_backend()
    executor, evalr = make_executor(repo_root, runs_dir, backend, logs)

    assert executor.run() == 0

    for name in ("r_s0", "r_s1"):
        run_dir = runs_dir / name
        for output in TRAIN_OUTPUTS:
            assert (run_dir / output).exists()
        assert (run_dir / "eval.json").exists()
        assert json.loads(
            (run_dir / "backend.json").read_text(encoding="utf-8")
        ) == {
            "finished": ISO_NOW,
            "gpu": "fake",
            "kind": "fake",
            "plan_id": "p1",
            "started": ISO_NOW,
        }

    for name in ("r2y_s0", "r2y_s1"):
        out_dir = runs_dir / name
        assert (out_dir / "test_scores.jsonl").exists()
        assert (out_dir / "eval.json").exists()
        assert (out_dir / "backend.json").exists()

    assert evalr.argvs == [
        [
            "uv", "run", "python", "-m", "training.eval_samples",
            "--samples", str(runs_dir / "r_s0" / "samples.json"),
            "--vocab", "data/tok_x/vocab.json",
            "--train-tokens", "data/tok_x/train.jsonl",
            "--out", str(runs_dir / "r_s0" / "eval.json"),
        ],
        [
            "uv", "run", "python", "-m", "training.eval_samples",
            "--samples", str(runs_dir / "r_s1" / "samples.json"),
            "--vocab", "data/tok_x/vocab.json",
            "--train-tokens", "data/tok_x/train.jsonl",
            "--out", str(runs_dir / "r_s1" / "eval.json"),
        ],
    ]

    meaningful = [call for call in backend.calls if call[0] != "alive"]
    assert meaningful[0] == ("acquire",)
    assert meaningful[1] == (
        "put", str(repo_root / "training" / "train_ar.py"), "/fake/train_ar.py",
    )
    assert meaningful[2] == (
        "put", str(repo_root / "training" / "grammar_mask.py"),
        "/fake/grammar_mask.py",
    )
    assert tuple(meaningful[3:10]) == (
        ("put", str(repo_root / "data" / "tok_x" / "train.jsonl"), "/fake/train.jsonl"),
        ("put", str(repo_root / "data" / "tok_x" / "val.jsonl"), "/fake/val.jsonl"),
        ("put", str(repo_root / "data" / "tok_x" / "test.jsonl"), "/fake/test.jsonl"),
        ("put", str(repo_root / "data" / "tok_x" / "vocab.json"), "/fake/vocab.json"),
        ("put", str(repo_root / "data" / "tok_x" / "meta.json"), "/fake/meta.json"),
        ("put", str(repo_root / "data" / "tok_x" / "vocab.json"), "/fake/vocab_tok_x.json"),
        ("put", str(repo_root / "data" / "tok_x" / "meta.json"), "/fake/meta_tok_x.json"),
    )

    assert "=== r_s0 start 12:00:00 ===" in logs
    assert "=== r_s0 done 12:00:00 ===" in logs
    assert "=== r_s1 start 12:00:00 ===" in logs
    assert "=== r_s1 done 12:00:00 ===" in logs
    assert "=== rescore tok_x 12:00:00 ===" in logs
    assert "RUN-DONE 12:00:00" in logs


# --- idempotency -------------------------------------------------------------

def test_completed_train_job_is_skipped(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    run_dir = runs_dir / "r_s0"
    run_dir.mkdir()
    (run_dir / "test_scores.jsonl").write_text("x\n", encoding="utf-8")
    (run_dir / "model.pt").write_bytes(b"model")
    (run_dir / "samples.json").write_text("{}", encoding="utf-8")
    logs: list[str] = []
    backend = make_backend()
    executor, _ = make_executor(repo_root, runs_dir, backend, logs)

    assert executor.run() == 0
    assert "skip r_s0" in logs
    assert not any(
        call[0] == "run" and call[1] == "r_s0" for call in backend.calls
    )
    assert (runs_dir / "r2y_s0" / "test_scores.jsonl").exists()


def test_complete_block_is_skipped_without_staging(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    for name in ("r_s0", "r_s1", "r2y_s0", "r2y_s1"):
        run_dir = runs_dir / name
        run_dir.mkdir()
        (run_dir / "test_scores.jsonl").write_text("x\n", encoding="utf-8")
    logs: list[str] = []
    backend = make_backend()
    executor, evalr = make_executor(repo_root, runs_dir, backend, logs)

    assert executor.run() == 0
    assert "skip data/tok_x (complete)" in logs
    puts = [call for call in backend.calls if call[0] == "put"]
    assert [call[2] for call in puts] == ["/fake/train_ar.py", "/fake/grammar_mask.py"]
    assert evalr.argvs == []
    assert not any(call[0] == "run" for call in backend.calls)


# --- failure handling --------------------------------------------------------

def test_partial_run_is_removed_and_block_retried(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    backend = make_backend(omit_r_s1_first=True)
    executor, _ = make_executor(repo_root, runs_dir, backend, logs)

    assert executor.run() == 0
    assert "=== r_s1 FAILED 12:00:00 ===" in logs
    for output in TRAIN_OUTPUTS:
        assert (runs_dir / "r_s1" / output).exists()
    assert (runs_dir / "r_s1" / "eval.json").exists()
    assert (runs_dir / "r_s1" / "backend.json").exists()


def test_two_failures_return_3_and_release(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    backend = make_backend(omit_r_s1_always=True)
    executor, _ = make_executor(repo_root, runs_dir, backend, logs)

    assert executor.run() == 3
    assert "=== r_s1 FAILED 12:00:00 ===" in logs
    assert not (runs_dir / "r_s1").exists()
    assert ("release",) in backend.calls


def test_session_lost_reacquires_and_resumes(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    created: list[FakeBackend] = []

    def factory() -> FakeBackend:
        backend = make_backend()
        if not created:
            backend.run_errors["r_s0"] = SessionLost("gone")
        created.append(backend)
        return backend

    evalr = RecordingEval()
    router = Router({"colab": factory}, "colab", log=logs.append)
    executor = PlanExecutor(
        PLAN, router, repo_root=repo_root, runs_dir=runs_dir,
        local_eval=evalr, log=logs.append, now=lambda: NOW,
    )

    assert executor.run() == 0
    assert "SESSION-LOST at data/tok_x" in logs
    assert len(created) == 2
    code_puts = [
        call
        for backend in created
        for call in backend.calls
        if call[0] == "put"
        and call[2] in ("/fake/train_ar.py", "/fake/grammar_mask.py")
    ]
    assert len(code_puts) == 4
    assert (runs_dir / "r_s0" / "test_scores.jsonl").exists()
    assert (runs_dir / "r2y_s0" / "test_scores.jsonl").exists()


def test_backend_unavailable_on_reacquire_returns_2(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    first = make_backend()
    first.run_errors["r_s0"] = SessionLost("gone")
    second = FakeBackend(acquire_failures=1)
    second.kind = "colab"
    state = {"calls": 0}

    def factory() -> FakeBackend:
        state["calls"] += 1
        return first if state["calls"] == 1 else second

    evalr = RecordingEval()
    router = Router({"colab": factory}, "colab", log=logs.append)
    executor = PlanExecutor(
        PLAN, router, repo_root=repo_root, runs_dir=runs_dir,
        local_eval=evalr, log=logs.append, now=lambda: NOW,
    )

    assert executor.run() == 2
    assert ("release",) not in first.calls


def test_missing_rescore_output_returns_3(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    backend = make_backend(omit_rescore=True)
    executor, _ = make_executor(repo_root, runs_dir, backend, logs)

    assert executor.run() == 3
    assert not (runs_dir / "r2y_s0").exists()
    assert ("release",) in backend.calls
    assert (
        "FAILURE at data/tok_x: missing rescore output: r2y_s0, r2y_s1"
        in logs
    )


# --- dry run -----------------------------------------------------------------

def test_dry_run_logs_and_touches_nothing(tmp_path: Path) -> None:
    repo_root, runs_dir = build_repo(tmp_path)
    logs: list[str] = []
    backend = make_backend()
    factory_calls: list[int] = []

    def factory() -> FakeBackend:
        factory_calls.append(1)
        return backend

    evalr = RecordingEval()
    router = Router({"colab": factory}, "colab", log=logs.append)
    executor = PlanExecutor(
        PLAN, router, repo_root=repo_root, runs_dir=runs_dir,
        local_eval=evalr, log=logs.append, dry_run=True, now=lambda: NOW,
    )

    assert executor.run() == 0
    assert list(runs_dir.iterdir()) == []
    assert factory_calls == []
    assert evalr.argvs == []

    joined = "\n".join(logs)
    assert "PUT " in joined
    assert "GET " in joined
    assert "RUN r_s0 timeout=6000" in logs
    assert any(line.startswith("EVAL ") for line in logs)
    assert "RUN-DONE 12:00:00" in logs
    index = logs.index("RUN r_s0 timeout=6000")
    assert logs[index + 1] == train_wrapper(PLAN.blocks[0].train[0], "/content")
