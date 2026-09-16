"""Backend protocol and the injectable command-runner seam.

An experiment ``Plan`` is execution-agnostic; a ``Backend`` is the machine that
trains it. This module defines that contract, its failure exceptions, the
injectable ``CommandRunner`` callable used by the concrete backends, and the
default ``run_command`` subprocess implementation. Standard library only.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class BackendUnavailable(RuntimeError):
    """Acquisition failed (no allocation, GPU busy)."""


class SessionLost(RuntimeError):
    """The execution target disappeared after acquisition."""


class RemoteFailure(RuntimeError):
    """The target is alive but a command failed."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


CommandRunner = Callable[[Sequence[str], int], CommandResult]

OUTPUT_FILTER: re.Pattern[str] = re.compile(
    r"early stop|test NLL|Error|Traceback|lost|RESCORED|RESCORE-DONE"
)


def filter_output(text: str) -> str:
    """Keep only the lines matching ``OUTPUT_FILTER`` (no trailing newline)."""
    return "\n".join(
        line for line in text.splitlines() if OUTPUT_FILTER.search(line)
    )


def run_command(argv: Sequence[str], timeout_s: float) -> CommandResult:
    """Run ``argv``, capturing stdout and stderr; timeouts return code 124."""
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(124, f"timeout after {timeout_s}s")
    return CommandResult(completed.returncode, completed.stdout + completed.stderr)


class Backend(Protocol):
    kind: str
    root: str

    def acquire(self) -> None: ...

    def release(self) -> None: ...

    def alive(self) -> bool: ...

    def put(self, local: Path, remote: str) -> None: ...

    def get(self, remote: str, local: Path) -> bool: ...

    def run(self, script: str, name: str, timeout_s: int) -> str: ...

    def describe(self) -> dict[str, object]: ...
