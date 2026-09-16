"""Backend policy resolution, acquisition and re-acquisition.

The policy names where a Plan should run. It is chosen from the command line,
then the ``GR_BACKEND`` environment variable, then a default (``colab``). The
Router turns the policy into a concrete ``Backend`` and replaces it when the
session is lost. Standard library only: no torch, no cfg_reducer.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from .backend import Backend, BackendUnavailable

DEFAULT_POLICY = "colab"
POLICIES = ("colab", "local", "auto")
KINDS = ("colab", "local")


def resolve_policy(cli: str | None, env: Mapping[str, str] | None = None) -> str:
    """Return the effective policy: cli, else ``GR_BACKEND``, else the default."""
    if env is None:
        env = os.environ
    if cli is not None:
        policy = cli
    else:
        from_env = env.get("GR_BACKEND")
        policy = from_env if from_env else DEFAULT_POLICY
    if policy not in POLICIES:
        raise ValueError(f"unknown backend policy {policy!r}")
    return policy


class Router:
    """Choose and (re-)acquire a backend according to a policy."""

    def __init__(
        self,
        factories: Mapping[str, Callable[[], Backend]],
        policy: str,
        *,
        allow_switch: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown backend policy {policy!r}")
        self.factories = dict(factories)
        self.policy = policy
        self.allow_switch = allow_switch
        self.log = log

    def order(self) -> tuple[str, ...]:
        if self.policy in KINDS:
            return (self.policy,)
        other = next(kind for kind in KINDS if kind != DEFAULT_POLICY)
        return (DEFAULT_POLICY, other)

    def acquire(self) -> Backend:
        for kind in self.order():
            factory = self.factories.get(kind)
            if factory is None:
                self.log(f"no factory for {kind}")
                continue
            backend = factory()
            try:
                backend.acquire()
            except BackendUnavailable as exc:
                self.log(f"backend {kind} unavailable: {exc}")
                continue
            return backend
        raise BackendUnavailable("no backend available")

    def reacquire(self, previous: Backend) -> Backend:
        kind = previous.kind
        factory = self.factories.get(kind)
        if factory is None:
            self.log(f"no factory for {kind}")
            failure: BackendUnavailable = BackendUnavailable(
                f"no factory for {kind}"
            )
        else:
            try:
                backend = factory()
                backend.acquire()
            except BackendUnavailable as exc:
                self.log(f"backend {kind} unavailable: {exc}")
                failure = exc
            else:
                return backend
        if not self.allow_switch:
            raise failure
        for other in KINDS:
            if other == kind:
                continue
            other_factory = self.factories.get(other)
            if other_factory is None:
                self.log(f"no factory for {other}")
                continue
            try:
                backend = other_factory()
                backend.acquire()
            except BackendUnavailable as exc:
                self.log(f"backend {other} unavailable: {exc}")
                continue
            return backend
        raise BackendUnavailable("no backend available")
