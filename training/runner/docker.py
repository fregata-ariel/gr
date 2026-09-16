"""Local Docker backend: run Plans on the local GPU inside a container.

Follows the same put/run/get contract as the Colab adapter so the executor is
identical for both targets. Code and data are staged into a directory that is
bind-mounted at ``/work``; ``GR_ROOT=/work`` tells ``train_ar`` where to look.
A non-blocking file lock guarantees the local GPU is owned by one Plan at a
time. Standard library only: no torch, no cfg_reducer.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

from .backend import (
    BackendUnavailable,
    CommandResult,
    CommandRunner,
    RemoteFailure,
    filter_output,
    run_command,
)

GPU_PROBE_TIMEOUT = 120
IMAGE_INSPECT_TIMEOUT = 60
FAILURE_TAIL_LINES = 40


class DockerBackend:
    kind = "local"
    root = "/work"

    def __init__(
        self,
        image: str,
        staging: Path,
        *,
        gpus: str = "all",
        run_command: CommandRunner = run_command,
        lock_path: Path = Path.home() / ".cache" / "gr-runner" / "local-gpu.lock",
        uid_gid: str | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.image = image
        self.staging = Path(staging).resolve()
        self.gpus = gpus
        self.lock_path = lock_path
        self.uid_gid = (
            uid_gid if uid_gid is not None else f"{os.getuid()}:{os.getgid()}"
        )
        self.gpu_name = ""
        self.image_id = ""
        self._lock_file: IO[str] | None = None
        self._run_command = run_command
        self.log = log

    @staticmethod
    def _first_nonempty_line(text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return ""

    @staticmethod
    def _now() -> str:
        return time.strftime("%H:%M:%S")

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.lock_path, "a", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_file.close()
            raise BackendUnavailable("local GPU is in use")
        self._lock_file = lock_file

        probe = self._run_command(
            [
                "docker", "run", "--rm", "--gpus", self.gpus, self.image,
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            ],
            GPU_PROBE_TIMEOUT,
        )
        if probe.returncode != 0:
            self.release()
            raise BackendUnavailable(probe.output)
        self.gpu_name = self._first_nonempty_line(probe.output)

        inspect = self._run_command(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
            IMAGE_INSPECT_TIMEOUT,
        )
        self.image_id = inspect.output.strip() if inspect.returncode == 0 else ""

        self.staging.mkdir(parents=True, exist_ok=True)
        self.log(f"SESSION-OK {self._now()}")

    def release(self) -> None:
        lock_file = self._lock_file
        if lock_file is None:
            return
        self._lock_file = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def alive(self) -> bool:
        return True

    def _staged(self, remote: str) -> Path:
        if remote != self.root and not remote.startswith(self.root + "/"):
            raise ValueError(f"remote path {remote!r} is outside {self.root!r}")
        return self.staging / remote[len(self.root) + 1:]

    def put(self, local: Path, remote: str) -> None:
        destination = self._staged(remote)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, destination)

    def get(self, remote: str, local: Path) -> bool:
        source = self._staged(remote)
        if not source.exists():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)
        return True

    def run(self, script: str, name: str, timeout_s: int) -> str:
        scripts_dir = self.staging / "_scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"{name}.py"
        script_path.write_text(script, encoding="utf-8")
        result: CommandResult = self._run_command(
            [
                "docker", "run", "--rm", "--gpus", self.gpus,
                "--user", self.uid_gid,
                "-e", "HOME=/tmp",
                "-e", "GR_ROOT=/work",
                "-v", f"{self.staging}:/work",
                self.image,
                "python", f"/work/_scripts/{name}.py",
            ],
            timeout_s,
        )
        if result.returncode != 0:
            tail = "\n".join(result.output.splitlines()[-FAILURE_TAIL_LINES:])
            raise RemoteFailure(tail)
        return filter_output(result.output)

    def describe(self) -> dict[str, object]:
        return {
            "gpu": self.gpu_name or "",
            "image": self.image,
            "image_id": self.image_id or "",
            "kind": "local",
        }
