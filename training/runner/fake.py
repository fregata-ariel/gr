"""In-process test doubles for the runner executor tests.

``FakeBackend`` records every call, serves files from an in-memory dict, and
lets a test script the acquisition failures, per-run errors, liveness sequence
and fabricated outputs. ``DryRunBackend`` only logs the command sequence so
``--dry-run`` can show what would happen without any target being touched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .backend import BackendUnavailable


class FakeBackend:
    kind = "fake"
    root = "/fake"

    def __init__(
        self,
        *,
        acquire_failures: int = 0,
        on_run: Callable[[str, str], str] | None = None,
        run_errors: dict[str, Exception] | None = None,
        alive_sequence: Iterable[bool] = (),
    ) -> None:
        self.acquire_failures = acquire_failures
        self.on_run = on_run
        self.run_errors = dict(run_errors or {})
        self.alive_sequence = list(alive_sequence)
        self.calls: list[tuple[str, ...]] = []
        self.files: dict[str, bytes] = {}
        self.scripts: dict[str, str] = {}
        self._acquire_calls = 0

    def acquire(self) -> None:
        self.calls.append(("acquire",))
        self._acquire_calls += 1
        if self._acquire_calls <= self.acquire_failures:
            raise BackendUnavailable(
                f"fake acquire failure {self._acquire_calls}"
            )

    def release(self) -> None:
        self.calls.append(("release",))

    def alive(self) -> bool:
        self.calls.append(("alive",))
        if self.alive_sequence:
            return self.alive_sequence.pop(0)
        return True

    def put(self, local: Path, remote: str) -> None:
        self.calls.append(("put", str(local), remote))
        self.files[remote] = Path(local).read_bytes()

    def get(self, remote: str, local: Path) -> bool:
        self.calls.append(("get", remote, str(local)))
        if remote not in self.files:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.files[remote])
        return True

    def run(self, script: str, name: str, timeout_s: int) -> str:
        self.calls.append(("run", name, str(timeout_s)))
        self.scripts[name] = script
        if name in self.run_errors:
            raise self.run_errors.pop(name)
        if self.on_run is not None:
            return self.on_run(name, script)
        return ""

    def describe(self) -> dict[str, object]:
        return {"gpu": "fake", "kind": "fake"}


class DryRunBackend:
    kind = "dry"

    def __init__(
        self,
        root: str = "/content",
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        self.root = root
        self.log = log

    def acquire(self) -> None:
        self.log("ACQUIRE dry")

    def release(self) -> None:
        self.log("RELEASE dry")

    def alive(self) -> bool:
        return True

    def put(self, local: Path, remote: str) -> None:
        self.log(f"PUT {local} -> {remote}")

    def get(self, remote: str, local: Path) -> bool:
        self.log(f"GET {remote} -> {local}")
        return True

    def run(self, script: str, name: str, timeout_s: int) -> str:
        self.log(f"RUN {name} timeout={timeout_s}")
        self.log(script)
        return ""

    def describe(self) -> dict[str, object]:
        return {"kind": "dry"}
