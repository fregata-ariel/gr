"""Tests for the local Docker backend, the pinned Dockerfile and GR_ROOT.

A fake ``CommandRunner`` records every ``(argv, timeout)`` pair and returns
scripted results; staging and the lock file live in ``tmp_path``. The real
``docker`` tool is never invoked.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from training.runner import (
    BackendUnavailable,
    CommandResult,
    DockerBackend,
    RemoteFailure,
)

IMAGE = "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeRunner:
    """Records calls and returns scripted results (default: returncode 0)."""

    def __init__(self, results: Sequence[CommandResult] | None = None) -> None:
        self.results: list[CommandResult] = list(results or [])
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def __call__(self, argv: Sequence[str], timeout_s: int) -> CommandResult:
        self.calls.append((tuple(argv), timeout_s))
        if self.results:
            return self.results.pop(0)
        return CommandResult(0, "")

    @property
    def argvs(self) -> list[tuple[str, ...]]:
        return [argv for argv, _ in self.calls]


def _noop_log(line: str) -> None:
    del line


def make_backend(
    runner: FakeRunner,
    tmp_path: Path,
    *,
    uid_gid: str | None = None,
    staging: Path | None = None,
    lock_path: Path | None = None,
    logs: list[str] | None = None,
) -> DockerBackend:
    return DockerBackend(
        IMAGE,
        staging if staging is not None else tmp_path / "staging",
        run_command=runner,
        lock_path=(
            lock_path
            if lock_path is not None
            else tmp_path / "cache" / "local-gpu.lock"
        ),
        uid_gid=uid_gid,
        log=logs.append if logs is not None else _noop_log,
    )


# --- acquire -----------------------------------------------------------------

def test_acquire_probes_gpu_and_inspects_image(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(0, "\nNVIDIA GeForce RTX 4090\n"),
        CommandResult(0, "sha256:abc\n"),
    ])
    logs: list[str] = []
    backend = make_backend(runner, tmp_path, logs=logs)

    backend.acquire()

    assert runner.calls == [
        (
            (
                "docker", "run", "--rm", "--gpus", "all", IMAGE,
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            ),
            120,
        ),
        (
            ("docker", "image", "inspect", "--format", "{{.Id}}", IMAGE),
            60,
        ),
    ]
    assert backend.gpu_name == "NVIDIA GeForce RTX 4090"
    assert backend.image_id == "sha256:abc"
    assert backend.staging.is_dir()
    assert logs[-1].startswith("SESSION-OK ")


def test_acquire_missing_image_id_is_empty(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(0, "GPU0\n"),
        CommandResult(1, "Error: no such image\n"),
    ])
    backend = make_backend(runner, tmp_path)

    backend.acquire()

    assert backend.gpu_name == "GPU0"
    assert backend.image_id == ""


def test_failing_probe_releases_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache" / "local-gpu.lock"
    backend_one = make_backend(
        FakeRunner([CommandResult(1, "could not select device driver\n")]),
        tmp_path,
        staging=tmp_path / "staging-one",
        lock_path=lock_path,
    )

    with pytest.raises(BackendUnavailable) as excinfo:
        backend_one.acquire()
    assert "could not select device driver" in str(excinfo.value)

    runner_two = FakeRunner([CommandResult(0, "GPU1\n"), CommandResult(0, "id\n")])
    backend_two = make_backend(
        runner_two,
        tmp_path,
        staging=tmp_path / "staging-two",
        lock_path=lock_path,
    )
    backend_two.acquire()

    assert backend_two.gpu_name == "GPU1"


def test_second_acquire_fails_while_locked(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache" / "local-gpu.lock"
    backend_one = make_backend(
        FakeRunner([CommandResult(0, "GPU0\n"), CommandResult(0, "id\n")]),
        tmp_path,
        staging=tmp_path / "staging-one",
        lock_path=lock_path,
    )
    backend_one.acquire()

    runner_two = FakeRunner([CommandResult(0, "GPU1\n"), CommandResult(0, "id\n")])
    backend_two = make_backend(
        runner_two,
        tmp_path,
        staging=tmp_path / "staging-two",
        lock_path=lock_path,
    )
    with pytest.raises(BackendUnavailable):
        backend_two.acquire()
    assert runner_two.calls == []

    backend_one.release()
    backend_two.acquire()
    assert backend_two.gpu_name == "GPU1"


def test_release_is_idempotent(tmp_path: Path) -> None:
    backend = make_backend(
        FakeRunner([CommandResult(0, "GPU\n"), CommandResult(0, "id\n")]),
        tmp_path,
    )
    backend.acquire()
    backend.release()
    backend.release()
    assert backend.alive() is True


# --- put / get ---------------------------------------------------------------

def test_put_get_roundtrip(tmp_path: Path) -> None:
    backend = make_backend(FakeRunner(), tmp_path)
    source = tmp_path / "train.jsonl"
    source.write_bytes(b"hello\n")

    backend.put(source, "/work/data/train.jsonl")

    staged = backend.staging / "data" / "train.jsonl"
    assert staged.read_bytes() == b"hello\n"

    destination = tmp_path / "out" / "copy.jsonl"
    assert backend.get("/work/data/train.jsonl", destination) is True
    assert destination.read_bytes() == b"hello\n"
    assert backend.get("/work/data/missing.jsonl", tmp_path / "nope") is False


def test_remote_outside_root_raises(tmp_path: Path) -> None:
    backend = make_backend(FakeRunner(), tmp_path)
    source = tmp_path / "x.txt"
    source.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        backend.put(source, "/other/x.txt")
    with pytest.raises(ValueError):
        backend.put(source, "/workx/x.txt")
    with pytest.raises(ValueError):
        backend.get("/other/x.txt", tmp_path / "y.txt")


# --- run ---------------------------------------------------------------------

def test_run_argv_script_and_filtering(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(0, "early stop\nnoise line\ntest NLL 1.23\nError: bad\n")
    ])
    backend = make_backend(runner, tmp_path, uid_gid="1000:1000")

    output = backend.run("print('hi')\n", "job", 1234)

    script_path = backend.staging / "_scripts" / "job.py"
    assert script_path.read_text(encoding="utf-8") == "print('hi')\n"
    assert runner.calls == [
        (
            (
                "docker", "run", "--rm", "--gpus", "all",
                "--user", "1000:1000",
                "-e", "HOME=/tmp",
                "-e", "GR_ROOT=/work",
                "-v", f"{backend.staging}:/work",
                IMAGE,
                "python", "/work/_scripts/job.py",
            ),
            1234,
        )
    ]
    assert output == "early stop\ntest NLL 1.23\nError: bad"


def test_run_remote_failure_tail(tmp_path: Path) -> None:
    output = "\n".join(f"line{i}" for i in range(50))
    runner = FakeRunner([CommandResult(1, output)])
    backend = make_backend(runner, tmp_path, uid_gid="1000:1000")

    with pytest.raises(RemoteFailure) as excinfo:
        backend.run("print('x')\n", "job", 60)

    text = str(excinfo.value)
    assert "line10" in text
    assert "line49" in text
    assert "line9" not in text


# --- describe ----------------------------------------------------------------

def test_describe_has_four_keys(tmp_path: Path) -> None:
    backend = make_backend(FakeRunner(), tmp_path)
    assert backend.kind == "local"
    assert backend.root == "/work"
    assert set(backend.describe()) == {"gpu", "image", "image_id", "kind"}

    backend = make_backend(
        FakeRunner([CommandResult(0, "GPU0\n"), CommandResult(0, "sha256:x\n")]),
        tmp_path,
    )
    backend.acquire()
    assert backend.describe() == {
        "gpu": "GPU0",
        "image": IMAGE,
        "image_id": "sha256:x",
        "kind": "local",
    }


# --- repository files --------------------------------------------------------

def test_dockerfile_has_only_pinned_from() -> None:
    text = (REPO_ROOT / "training" / "docker" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    instructions = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert instructions == [f"FROM {IMAGE}"]


def test_train_ar_uses_gr_root_defaults() -> None:
    text = (REPO_ROOT / "training" / "train_ar.py").read_text(encoding="utf-8")
    assert 'os.environ.get("GR_ROOT", "/content")' in text
    assert 'default="/content/' not in text
