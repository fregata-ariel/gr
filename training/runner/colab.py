"""Colab backend: an adapter around the ``colab`` command line tool.

Reproduces the existing shell runner exactly: the same argv, timeouts, GPU
allocation retry loop, session-loss detection, and strict serialisation of all
``colab`` commands through a file lock (``colab new`` racing another command
corrupts the session registry).
"""

from __future__ import annotations

import fcntl
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .backend import (
    BackendUnavailable,
    CommandResult,
    CommandRunner,
    RemoteFailure,
    SessionLost,
    filter_output,
    run_command,
)


@dataclass(frozen=True)
class ColabTimeouts:
    new: int = 120
    sessions: int = 60
    upload: int = 300
    download: int = 300
    exec_outer: int = 6000
    exec_inner: int = 5400
    stop: int = 90


class ColabBackend:
    kind = "colab"
    root = "/content"

    def __init__(
        self,
        session: str,
        gpu: str = "T4",
        *,
        run_command: CommandRunner = run_command,
        lock_path: Path = Path.home() / ".cache" / "gr-runner" / "colab.lock",
        attempts: int = 36,
        sleep_s: int = 600,
        timeouts: ColabTimeouts = ColabTimeouts(),
        sleep: Callable[[int], None] = time.sleep,
        log: Callable[[str], None] = print,
        script_dir: Path | None = None,
    ) -> None:
        self.session = session
        self.gpu = gpu
        self.lock_path = lock_path
        self.attempts = attempts
        self.sleep_s = sleep_s
        self.timeouts = timeouts
        self.script_dir = (
            Path(tempfile.mkdtemp(prefix="gr-runner-"))
            if script_dir is None
            else script_dir
        )
        self._run_command: CommandRunner = run_command
        self._sleep = sleep
        self.log = log

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _run_locked(self, argv: Sequence[str], timeout_s: int) -> CommandResult:
        with self._lock():
            return self._run_command(argv, timeout_s)

    @staticmethod
    def _last_line(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _session_listed(self, output: str) -> bool:
        prefix = f"[{self.session}]"
        return any(line.startswith(prefix) for line in output.splitlines())

    def _raise_for_failure(self, output: str) -> None:
        if self.alive():
            raise RemoteFailure(output)
        raise SessionLost(output)

    @staticmethod
    def _now() -> str:
        return time.strftime("%H:%M:%S")

    def acquire(self) -> None:
        for attempt in range(1, self.attempts + 1):
            with self._lock():
                new_result = self._run_command(
                    ["colab", "new", "--gpu", self.gpu, "-s", self.session],
                    self.timeouts.new,
                )
                sessions_result = self._run_command(
                    ["colab", "sessions"],
                    self.timeouts.sessions,
                )
            if self._session_listed(sessions_result.output):
                self.log(f"SESSION-OK {self._now()}")
                return
            last = self._last_line(new_result.output) or self._last_line(
                sessions_result.output
            )
            self.log(f"attempt {attempt}/{self.attempts} {self._now()}: {last}")
            if attempt < self.attempts:
                self._sleep(self.sleep_s)
        self.log(f"GIVE-UP {self._now()}")
        raise BackendUnavailable(
            f"could not acquire session {self.session!r} after {self.attempts} attempts"
        )

    def release(self) -> None:
        result = self._run_locked(
            ["colab", "stop", "-s", self.session],
            self.timeouts.stop,
        )
        if result.returncode != 0:
            self.log(f"stop failed: {self._last_line(result.output)}")

    def alive(self) -> bool:
        result = self._run_locked(
            ["colab", "sessions"],
            self.timeouts.sessions,
        )
        return self._session_listed(result.output)

    def put(self, local: Path, remote: str) -> None:
        result = self._run_locked(
            ["colab", "upload", "-s", self.session, str(local), remote],
            self.timeouts.upload,
        )
        if result.returncode != 0:
            self._raise_for_failure(result.output)

    def get(self, remote: str, local: Path) -> bool:
        result = self._run_locked(
            ["colab", "download", "-s", self.session, remote, str(local)],
            self.timeouts.download,
        )
        if "not found" in result.output.lower():
            return False
        if result.returncode != 0:
            self._raise_for_failure(result.output)
        return True

    def run(self, script: str, name: str, timeout_s: int) -> str:
        path = self.script_dir / f"{name}.py"
        path.write_text(script, encoding="utf-8")
        result = self._run_locked(
            [
                "colab", "exec", "-s", self.session,
                "-f", str(path),
                "--timeout", str(self.timeouts.exec_inner),
            ],
            timeout_s if timeout_s else self.timeouts.exec_outer,
        )
        if result.returncode != 0 or "lost" in result.output.lower():
            self._raise_for_failure(result.output)
        return filter_output(result.output)

    def describe(self) -> dict[str, object]:
        return {"gpu": self.gpu, "kind": "colab", "session": self.session}
