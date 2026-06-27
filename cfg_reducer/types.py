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


# ──────────────────────────────────────────────
#  Motif — structural block extracted from Op history
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Motif:
    """
    A structural building block extracted by interpreting the
    reduction history in reverse (undo / reconstruction order).

    kind:
        "entry"   — root node with no predecessors
        "linear"  — single-predecessor extension
        "merge"   — multi-predecessor convergence (branch join)
        "loop"    — back-edge restoration completing a cycle

    node:   the node being reconstructed (None for loop motifs)
    preds:  predecessor node ids at reconstruction time
    succs:  successor node ids at reconstruction time
    meta:   extra info — for loops: {"header": ..., "scc": [...],
                                      "back_edges": [...]}
    step:   index in reconstruction order (0 = first undo replayed)
    """
    kind: str
    node: str | None
    preds: tuple[str, ...]
    succs: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    children: tuple['Motif', ...] = ()


@dataclass(frozen=True)
class MetaGraph:
    """
    DAG of Motif dependencies at one level of the hierarchy.

    motifs:     Motif nodes at this level (ordered by step).
    edges:      (src_step, dst_step) pairs — src restored a node that
                dst references in preds/succs.
    subgraphs:  Loop Motif step → MetaGraph of its children.
    """
    motifs: tuple[Motif, ...]
    edges: tuple[tuple[int, int], ...]
    subgraphs: dict[int, 'MetaGraph'] = field(default_factory=dict)
