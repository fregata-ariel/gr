"""
Pure data definitions.
These carry no behaviour and map directly to JSON — designed to be
trivially portable to Rust/C++ structs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────
#  NodeType
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class NodeType:
    """
    Classification of a node.
    default_weight is assigned at node creation and propagated
    upward during reverse-Kahn reduction.
    """
    name: str
    default_weight: int = 1


# The single built-in type.  More can be registered via
# engine.define_type() without touching this file.
BASIC = NodeType("basic", 1)


# ──────────────────────────────────────────────
#  Op  — atomic graph mutation record
# ──────────────────────────────────────────────

@dataclass
class Op:
    """
    Every graph mutation is recorded as an Op.

    kind     — handler key  ("remove_node" | "remove_edges" | custom)
    forward  — parameters for apply
    inverse  — parameters for revert  (local snapshot, not full copy)
    meta     — algorithm-level state snapshot (scope, etc.)
    """
    kind: str
    forward: dict[str, Any] = field(default_factory=dict)
    inverse: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
