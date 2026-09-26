"""Tests for the Colab backend and the command-runner seam.

A fake ``CommandRunner`` records every ``(argv, timeout)`` pair and returns
scripted results; sleep and log are captured in lists. The real ``colab`` tool
is never invoked.
"""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from training.runner import (
    BackendUnavailable,
    ColabBackend,
    ColabTimeouts,
    CommandResult,
    RemoteFailure,
    SessionLost,
    filter_output,
    run_command,
)


class FakeRunner:
    """Records calls and returns scripted results (default: returncode 0)."""

    def __init__(
        self,
        results: Sequence[CommandResult] | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.results: list[CommandResult] = list(results or [])
        self.calls: list[tuple[tuple[str, ...], int]] = []
        self.on_call = on_call

    def __call__(self, argv: Sequence[str], timeout_s: int) -> CommandResult:
        self.calls.append((tuple(argv), timeout_s))
        if self.on_call is not None:
            self.on_call()
        if self.results:
            return self.results.pop(0)
        return CommandResult(0, "")

    @property
    def argvs(self) -> list[tuple[str, ...]]:
        return [argv for argv, _ in self.calls]


def _noop_sleep(seconds: int) -> None:
    del seconds


def _noop_log(line: str) -> None:
    del line


def make_backend(
    runner: FakeRunner,
    tmp_path: Path,
    *,
    attempts: int = 36,
    sleep_s: int = 600,
    sleep: Callable[[int], None] | None = None,
    logs: list[str] | None = None,
    timeouts: ColabTimeouts | None = None,
    session: str = "sw",
) -> ColabBackend:
    return ColabBackend(
        session,
        "T4",
        run_command=runner,
        lock_path=tmp_path / "cache" / "colab.lock",
        attempts=attempts,
        sleep_s=sleep_s,
        timeouts=timeouts if timeouts is not None else ColabTimeouts(),
        sleep=sleep if sleep is not None else _noop_sleep,
        log=logs.append if logs is not None else _noop_log,
        script_dir=tmp_path,
    )


# --- exact argv and timeouts -------------------------------------------------

def test_new_and_sessions_argv(tmp_path: Path) -> None:
    runner = FakeRunner([CommandResult(0, ""), CommandResult(0, "[sw] Ready\n")])
    logs: list[str] = []
    backend = make_backend(runner, tmp_path, logs=logs)
    timeouts = ColabTimeouts()

    backend.acquire()

    assert runner.calls[0] == (
        ("colab", "new", "--gpu", "T4", "-s", "sw"),
        timeouts.new,
    )
    assert runner.calls[1] == (("colab", "sessions"), timeouts.sessions)
    assert len(logs) == 1
    assert logs[0].startswith("SESSION-OK ")


def test_upload_argv_and_timeout(tmp_path: Path) -> None:
    runner = FakeRunner()
    backend = make_backend(runner, tmp_path)

    backend.put(Path("local.txt"), "/content/remote.txt")

    assert runner.calls == [
        (
            ("colab", "upload", "-s", "sw", "local.txt", "/content/remote.txt"),
            ColabTimeouts().upload,
        )
    ]


def test_download_argv_and_timeout(tmp_path: Path) -> None:
    runner = FakeRunner()
    backend = make_backend(runner, tmp_path)

    assert backend.get("/content/remote.txt", Path("local.txt")) is True

    assert runner.calls == [
        (
            ("colab", "download", "-s", "sw", "/content/remote.txt", "local.txt"),
            ColabTimeouts().download,
        )
    ]


def test_exec_argv_and_timeout(tmp_path: Path) -> None:
    runner = FakeRunner([CommandResult(0, "test NLL 1.0\n")])
    backend = make_backend(runner, tmp_path)

    backend.run("print('hi')\n", "job", 1234)

    script_path = tmp_path / "job.py"
    assert script_path.read_text(encoding="utf-8") == "print('hi')\n"
    assert runner.calls == [
        (
            (
                "colab", "exec", "-s", "sw", "-f", str(script_path),
                "--timeout", str(ColabTimeouts().exec_inner),
            ),
            1234,
        )
    ]


def test_exec_defaults_to_outer_timeout(tmp_path: Path) -> None:
    runner = FakeRunner([CommandResult(0, "test NLL 1.0\n")])
    backend = make_backend(runner, tmp_path)

    backend.run("print('hi')\n", "job", 0)

    assert runner.calls[0][1] == ColabTimeouts().exec_outer


def test_stop_argv_and_timeout(tmp_path: Path) -> None:
    runner = FakeRunner()
    backend = make_backend(runner, tmp_path)

    backend.release()

    assert runner.calls == [
        (("colab", "stop", "-s", "sw"), ColabTimeouts().stop)
    ]


# --- acquire retry loop ------------------------------------------------------

def test_acquire_retries_then_succeeds(tmp_path: Path) -> None:
    results = [
        CommandResult(1, "Service Unavailable\n"), CommandResult(0, ""),
        CommandResult(1, "Service Unavailable\n"), CommandResult(0, ""),
        CommandResult(0, ""), CommandResult(0, "[sw] Ready\n"),
    ]
    runner = FakeRunner(results)
    sleeps: list[int] = []
    logs: list[str] = []
    backend = make_backend(
        runner, tmp_path, attempts=5, sleep_s=7,
        sleep=sleeps.append, logs=logs,
    )

    backend.acquire()

    assert sleeps == [7, 7]
    attempt_lines = [line for line in logs if line.startswith("attempt ")]
    assert len(attempt_lines) == 2
    assert attempt_lines[0].startswith("attempt 1/5 ")
    assert attempt_lines[0].endswith("Service Unavailable")
    assert attempt_lines[1].startswith("attempt 2/5 ")
    assert logs[-1].startswith("SESSION-OK ")


def test_acquire_gives_up(tmp_path: Path) -> None:
    results = [
        CommandResult(1, "Service Unavailable\n"), CommandResult(0, ""),
    ] * 3
    runner = FakeRunner(results)
    sleeps: list[int] = []
    logs: list[str] = []
    backend = make_backend(
        runner, tmp_path, attempts=3, sleep_s=11,
        sleep=sleeps.append, logs=logs,
    )

    with pytest.raises(BackendUnavailable):
        backend.acquire()

    assert sleeps == [11, 11]
    attempt_lines = [line for line in logs if line.startswith("attempt ")]
    assert len(attempt_lines) == 3
    assert logs[-1].startswith("GIVE-UP ")


# --- alive -------------------------------------------------------------------

def test_alive_parses_session_lines(tmp_path: Path) -> None:
    listed = FakeRunner([CommandResult(0, "junk\n[sw] Open\n[other] x\n")])
    assert make_backend(listed, tmp_path).alive() is True

    missing = FakeRunner([CommandResult(0, "[other] x\n[sw2] y\n")])
    assert make_backend(missing, tmp_path).alive() is False

    indented = FakeRunner([CommandResult(0, " [sw] nope\n")])
    assert make_backend(indented, tmp_path).alive() is False


# --- put / get failure classification ---------------------------------------

def test_put_session_lost(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(1, "upload failed\n"),
        CommandResult(0, "[other] x\n"),
    ])
    backend = make_backend(runner, tmp_path)

    with pytest.raises(SessionLost):
        backend.put(Path("local.txt"), "/content/remote.txt")


def test_put_remote_failure(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(1, "upload failed\n"),
        CommandResult(0, "[sw] Open\n"),
    ])
    backend = make_backend(runner, tmp_path)

    with pytest.raises(RemoteFailure) as excinfo:
        backend.put(Path("local.txt"), "/content/remote.txt")
    assert "upload failed" in str(excinfo.value)


def test_get_not_found_is_false(tmp_path: Path) -> None:
    runner = FakeRunner([CommandResult(1, "File or directory not found\n")])
    backend = make_backend(runner, tmp_path)

    assert backend.get("/content/missing", Path("local.txt")) is False


# --- run: script file, filtering, session loss -------------------------------

def test_run_filters_output(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(0, "early stop\nnoise line\ntest NLL 1.23\nError: bad\n")
    ])
    backend = make_backend(runner, tmp_path)

    output = backend.run("print('x')\n", "job", 60)

    assert output == "early stop\ntest NLL 1.23\nError: bad"
    assert (tmp_path / "job.py").read_text(encoding="utf-8") == "print('x')\n"
    assert runner.argvs[0][5] == str(tmp_path / "job.py")


def test_run_session_lost(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(0, "Connection was lost\n"),
        CommandResult(0, "[other] x\n"),
    ])
    backend = make_backend(runner, tmp_path)

    with pytest.raises(SessionLost):
        backend.run("print('x')\n", "job", 60)
    assert (tmp_path / "job.py").exists()


def test_run_remote_failure(tmp_path: Path) -> None:
    runner = FakeRunner([
        CommandResult(1, "Traceback: boom\n"),
        CommandResult(0, "[sw] Open\n"),
    ])
    backend = make_backend(runner, tmp_path)

    with pytest.raises(RemoteFailure):
        backend.run("print('x')\n", "job", 60)


def test_release_swallows_stop_failure(tmp_path: Path) -> None:
    runner = FakeRunner([CommandResult(1, "stop boom\n")])
    logs: list[str] = []
    backend = make_backend(runner, tmp_path, logs=logs)

    backend.release()

    assert logs == ["stop failed: stop boom"]


def test_describe(tmp_path: Path) -> None:
    backend = make_backend(FakeRunner(), tmp_path)
    assert backend.describe() == {"gpu": "T4", "kind": "colab", "session": "sw"}
    assert backend.kind == "colab"
    assert backend.root == "/content"


# --- locking -----------------------------------------------------------------

def test_lock_is_held_during_commands(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache" / "colab.lock"
    observed: list[bool] = []

    def on_call() -> None:
        with open(lock_path, "a", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                observed.append(isinstance(exc, BlockingIOError))
            else:
                observed.append(False)

    runner = FakeRunner([CommandResult(0, "[sw] Ready\n")], on_call=on_call)
    backend = make_backend(runner, tmp_path)

    assert backend.alive() is True
    assert observed == [True]


def test_session_commands_take_the_session_lock_not_the_global_one(tmp_path: Path) -> None:
    global_lock = tmp_path / "cache" / "colab.lock"
    observed: list[tuple[bool, bool]] = []

    def held(path: Path) -> bool:
        with open(path, "a", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return False

    runner = FakeRunner([CommandResult(0, "ok\n")], on_call=lambda: observed.append(
        (held(global_lock), held(tmp_path / "cache" / "colab-sw.lock"))))
    backend = make_backend(runner, tmp_path)
    backend.put(tmp_path / "x", "/content/x")
    assert backend.session_lock_path == tmp_path / "cache" / "colab-sw.lock"
    assert observed == [(False, True)]


# --- default run_command -----------------------------------------------------

def test_run_command_captures_echo() -> None:
    result = run_command(["echo", "hello"], 5)
    assert result.returncode == 0
    assert result.output == "hello\n"


def test_run_command_timeout() -> None:
    result = run_command(["sleep", "5"], 0.1)
    assert result.returncode == 124
    assert result.output == "timeout after 0.1s"


def test_filter_output_keeps_only_matching_lines() -> None:
    text = "early stop here\nplain\ntest NLL 0.5\nRESCORE-DONE\n"
    assert filter_output(text) == "early stop here\ntest NLL 0.5\nRESCORE-DONE"
    assert filter_output("nothing\n") == ""
