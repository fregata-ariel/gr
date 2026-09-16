"""Tests for policy resolution and backend (re-)acquisition.

No target is touched: only ``FakeBackend`` instances are created.
"""

from __future__ import annotations

import pytest

from training.runner import (
    BackendUnavailable,
    FakeBackend,
    Router,
    resolve_policy,
)


def _kind(backend: FakeBackend, kind: str) -> FakeBackend:
    backend.kind = kind
    return backend


# --- resolve_policy ----------------------------------------------------------

def test_resolve_cli_wins_over_env() -> None:
    assert resolve_policy("local", {"GR_BACKEND": "colab"}) == "local"
    assert resolve_policy("colab", {"GR_BACKEND": "local"}) == "colab"


def test_resolve_env_used_when_cli_is_none() -> None:
    assert resolve_policy(None, {"GR_BACKEND": "local"}) == "local"
    assert resolve_policy(None, {"GR_BACKEND": "auto"}) == "auto"


def test_resolve_empty_env_falls_back_to_default() -> None:
    assert resolve_policy(None, {}) == "colab"
    assert resolve_policy(None, {"GR_BACKEND": ""}) == "colab"


def test_resolve_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GR_BACKEND", raising=False)
    assert resolve_policy(None) == "colab"
    monkeypatch.setenv("GR_BACKEND", "local")
    assert resolve_policy(None) == "local"


def test_resolve_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        resolve_policy("bogus")
    with pytest.raises(ValueError):
        resolve_policy(None, {"GR_BACKEND": "bogus"})


# --- order -------------------------------------------------------------------

def test_order_for_each_policy() -> None:
    assert Router({}, "colab").order() == ("colab",)
    assert Router({}, "local").order() == ("local",)
    assert Router({}, "auto").order() == ("colab", "local")


# --- acquire -----------------------------------------------------------------

def test_acquire_returns_first_available_and_logs_failure() -> None:
    logs: list[str] = []
    colab = _kind(FakeBackend(acquire_failures=1), "colab")
    local = _kind(FakeBackend(), "local")
    router = Router(
        {"colab": lambda: colab, "local": lambda: local},
        "auto",
        log=logs.append,
    )

    assert router.acquire() is local
    assert logs == ["backend colab unavailable: fake acquire failure 1"]


def test_acquire_skips_missing_factory() -> None:
    logs: list[str] = []
    local = _kind(FakeBackend(), "local")
    router = Router({"local": lambda: local}, "auto", log=logs.append)

    assert router.acquire() is local
    assert logs == ["no factory for colab"]


def test_acquire_raises_when_all_fail() -> None:
    colab = _kind(FakeBackend(acquire_failures=1), "colab")
    local = _kind(FakeBackend(acquire_failures=1), "local")
    router = Router(
        {"colab": lambda: colab, "local": lambda: local},
        "auto",
        log=lambda line: None,
    )

    with pytest.raises(BackendUnavailable):
        router.acquire()


def test_acquire_raises_when_no_factories() -> None:
    router = Router({}, "auto", log=lambda line: None)
    with pytest.raises(BackendUnavailable):
        router.acquire()


# --- reacquire ---------------------------------------------------------------

def test_reacquire_builds_same_kind() -> None:
    previous = _kind(FakeBackend(), "colab")
    fresh = _kind(FakeBackend(), "colab")
    local = _kind(FakeBackend(), "local")
    router = Router(
        {"colab": lambda: fresh, "local": lambda: local},
        "local",
        allow_switch=True,
        log=lambda line: None,
    )

    assert router.reacquire(previous) is fresh


def test_reacquire_reraises_without_allow_switch() -> None:
    previous = _kind(FakeBackend(), "colab")
    fresh = _kind(FakeBackend(acquire_failures=1), "colab")
    router = Router(
        {"colab": lambda: fresh},
        "colab",
        log=lambda line: None,
    )

    with pytest.raises(BackendUnavailable):
        router.reacquire(previous)


def test_reacquire_allow_switch_falls_back_to_other_kind() -> None:
    logs: list[str] = []
    previous = _kind(FakeBackend(), "colab")
    colab = _kind(FakeBackend(acquire_failures=1), "colab")
    local = _kind(FakeBackend(), "local")
    router = Router(
        {"colab": lambda: colab, "local": lambda: local},
        "auto",
        allow_switch=True,
        log=logs.append,
    )

    assert router.reacquire(previous) is local
    assert logs == ["backend colab unavailable: fake acquire failure 1"]


def test_reacquire_raises_when_switch_fails() -> None:
    previous = _kind(FakeBackend(), "colab")
    colab = _kind(FakeBackend(acquire_failures=1), "colab")
    local = _kind(FakeBackend(acquire_failures=1), "local")
    router = Router(
        {"colab": lambda: colab, "local": lambda: local},
        "auto",
        allow_switch=True,
        log=lambda line: None,
    )

    with pytest.raises(BackendUnavailable):
        router.reacquire(previous)
