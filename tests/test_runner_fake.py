"""Tests for the FakeBackend / DryRunBackend test doubles.

No target is touched: FakeBackend works entirely from memory and DryRunBackend
only appends to a log list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.runner import BackendUnavailable, DryRunBackend, FakeBackend


# --- FakeBackend -------------------------------------------------------------

def test_records_calls_in_order(tmp_path: Path) -> None:
    backend = FakeBackend()
    source = tmp_path / "a.txt"
    source.write_bytes(b"A")
    destination = tmp_path / "b.txt"

    backend.acquire()
    backend.alive()
    backend.put(source, "/fake/a.txt")
    assert backend.get("/fake/a.txt", destination) is True
    backend.run("script\n", "job", 10)
    backend.release()

    assert backend.calls == [
        ("acquire",),
        ("alive",),
        ("put", str(source), "/fake/a.txt"),
        ("get", "/fake/a.txt", str(destination)),
        ("run", "job", "10"),
        ("release",),
    ]


def test_acquire_failures() -> None:
    backend = FakeBackend(acquire_failures=2)

    with pytest.raises(BackendUnavailable):
        backend.acquire()
    with pytest.raises(BackendUnavailable):
        backend.acquire()
    backend.acquire()

    assert backend.calls == [("acquire",), ("acquire",), ("acquire",)]


def test_alive_sequence_then_true() -> None:
    backend = FakeBackend(alive_sequence=[False, False])

    assert backend.alive() is False
    assert backend.alive() is False
    assert backend.alive() is True


def test_put_get_roundtrip(tmp_path: Path) -> None:
    backend = FakeBackend()
    source = tmp_path / "src.bin"
    source.write_bytes(b"\x00\x01\x02")

    backend.put(source, "/fake/data/x.bin")

    assert backend.files == {"/fake/data/x.bin": b"\x00\x01\x02"}
    destination = tmp_path / "nested" / "out.bin"
    assert backend.get("/fake/data/x.bin", destination) is True
    assert destination.read_bytes() == b"\x00\x01\x02"
    assert backend.get("/fake/missing", tmp_path / "m.bin") is False


def test_run_errors_raised_once() -> None:
    backend = FakeBackend(run_errors={"job": RuntimeError("boom")})

    with pytest.raises(RuntimeError, match="boom"):
        backend.run("s\n", "job", 5)

    assert backend.run("s\n", "job", 5) == ""
    assert backend.scripts == {"job": "s\n"}


def test_on_run_can_fabricate_files(tmp_path: Path) -> None:
    def on_run(name: str, script: str) -> str:
        backend.files[f"/fake/{name}.json"] = b"out"
        return "filtered"

    backend = FakeBackend(on_run=on_run)

    assert backend.run("s\n", "job", 5) == "filtered"
    assert backend.get("/fake/job.json", tmp_path / "out.json") is True
    assert (tmp_path / "out.json").read_bytes() == b"out"


def test_describe() -> None:
    backend = FakeBackend()
    assert backend.kind == "fake"
    assert backend.root == "/fake"
    assert backend.describe() == {"gpu": "fake", "kind": "fake"}


# --- DryRunBackend -----------------------------------------------------------

def test_dry_run_logs_and_writes_nothing(tmp_path: Path) -> None:
    logs: list[str] = []
    backend = DryRunBackend("/dry", log=logs.append)
    local = tmp_path / "a.txt"
    destination = tmp_path / "b.txt"

    backend.acquire()
    backend.put(local, "/dry/a.txt")
    assert backend.get("/dry/a.txt", destination) is True
    backend.run("print(1)\n", "job", 60)
    backend.release()

    assert logs == [
        "ACQUIRE dry",
        f"PUT {local} -> /dry/a.txt",
        f"GET /dry/a.txt -> {destination}",
        "RUN job timeout=60",
        "print(1)\n",
        "RELEASE dry",
    ]
    assert not destination.exists()
    assert backend.alive() is True
    assert backend.describe() == {"kind": "dry"}
    assert backend.kind == "dry"


def test_dry_run_default_root() -> None:
    assert DryRunBackend(log=lambda line: None).root == "/content"
