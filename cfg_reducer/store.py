"""
Serialise / deserialise Op history to JSON.

The schema is deliberately flat so that a Rust/C++ reader
can consume the same files without a Python dependency.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .types import Op


# ──────────────────────────────────────────────
#  Codec helpers
# ──────────────────────────────────────────────

def _encode_value(v: Any) -> Any:
    """Convert sets to sorted lists for JSON serialisation."""
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, dict):
        return {k: _encode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_encode_value(item) for item in v]
    if isinstance(v, tuple):
        return [_encode_value(item) for item in v]
    return v


def _decode_op(d: dict) -> Op:
    return Op(
        kind=d["kind"],
        forward=d.get("forward", {}),
        inverse=d.get("inverse", {}),
        meta=d.get("meta", {}),
    )


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def save(ops: list[Op], path: str | Path) -> None:
    """Write a list of Ops to a JSON file."""
    payload = [
        {
            "kind": op.kind,
            "forward": _encode_value(op.forward),
            "inverse": _encode_value(op.inverse),
            "meta": _encode_value(op.meta),
        }
        for op in ops
    ]
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(path: str | Path) -> list[Op]:
    """Read a list of Ops from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_decode_op(d) for d in raw]
