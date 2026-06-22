"""
Graph engine — owns the adjacency structure, executes / reverts Ops,
and exposes a plugin registry for custom Op kinds.

The only module that touches mutable graph state.
Designed so that a future Rust/C++ rewrite replaces *only* this file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from .types import NodeType, Op, BASIC


# ──────────────────────────────────────────────
#  Node
# ──────────────────────────────────────────────

@dataclass
class Node:
    id: str
    succ: set[str] = field(default_factory=set)
    pred: set[str] = field(default_factory=set)
    weight: int = 1
    node_type: str = "basic"


# ──────────────────────────────────────────────
#  Op handler type
# ──────────────────────────────────────────────
#  apply_fn(engine, params) -> None
#  revert_fn(engine, params) -> None

ApplyFn = Callable[["GraphEngine", dict], None]
RevertFn = Callable[["GraphEngine", dict], None]


# ──────────────────────────────────────────────
#  Built-in handlers (module-level for clarity)
# ──────────────────────────────────────────────

def _apply_remove_node(engine: GraphEngine, params: dict) -> None:
    target = params["target"]
    node = engine.nodes[target]

    # Detach from predecessors
    for p in node.pred:
        if p in engine.nodes:
            engine.nodes[p].succ.discard(target)
            engine.nodes[p].weight += node.weight   # propagate weight

    # Detach from successors
    for s in node.succ:
        if s in engine.nodes:
            engine.nodes[s].pred.discard(target)

    del engine.nodes[target]


def _revert_remove_node(engine: GraphEngine, params: dict) -> None:
    target = params["target"]

    # Restore node
    engine.nodes[target] = Node(
        id=target,
        succ=set(params["succ_edges"]),
        pred=set(params["pred_edges"]),
        weight=params["weight"],
        node_type=params["node_type"],
    )

    # Reconnect predecessors and undo weight propagation
    weight_deltas: dict[str, int] = params["weight_deltas"]
    for p in params["pred_edges"]:
        if p in engine.nodes:
            engine.nodes[p].succ.add(target)
            engine.nodes[p].weight -= weight_deltas.get(p, 0)

    # Reconnect successors
    for s in params["succ_edges"]:
        if s in engine.nodes:
            engine.nodes[s].pred.add(target)


def _apply_remove_edges(engine: GraphEngine, params: dict) -> None:
    for src, dst in params["edges"]:
        if src in engine.nodes:
            engine.nodes[src].succ.discard(dst)
        if dst in engine.nodes:
            engine.nodes[dst].pred.discard(src)


def _revert_remove_edges(engine: GraphEngine, params: dict) -> None:
    for src, dst in params["edges"]:
        if src in engine.nodes:
            engine.nodes[src].succ.add(dst)
        if dst in engine.nodes:
            engine.nodes[dst].pred.add(src)


# ──────────────────────────────────────────────
#  GraphEngine
# ──────────────────────────────────────────────

class GraphEngine:
    """
    Thin wrapper around an adjacency-list directed graph.
    All mutations go through Op execution so they are undoable.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.type_registry: dict[str, NodeType] = {"basic": BASIC}

        # Op handler plugin table
        self._handlers: dict[str, tuple[ApplyFn, RevertFn]] = {}

        # Undo / redo stack
        self._history: list[Op] = []
        self._cursor: int = 0          # index of next empty slot

        # Register built-in handlers
        self.register("remove_node", _apply_remove_node, _revert_remove_node)
        self.register("remove_edges", _apply_remove_edges, _revert_remove_edges)

    # ── type management ──────────────────────

    def define_type(self, node_type: NodeType) -> None:
        self.type_registry[node_type.name] = node_type

    # ── graph construction (pre-reduction) ───

    def add_node(self, node_id: str, node_type: str = "basic") -> None:
        nt = self.type_registry[node_type]
        self.nodes[node_id] = Node(
            id=node_id,
            weight=nt.default_weight,
            node_type=node_type,
        )

    def add_edge(self, src: str, dst: str) -> None:
        self.nodes[src].succ.add(dst)
        self.nodes[dst].pred.add(src)

    # ── queries (read-only) ──────────────────

    def out_degree(self, nid: str) -> int:
        return len(self.nodes[nid].succ)

    def in_degree(self, nid: str) -> int:
        return len(self.nodes[nid].pred)

    def predecessors(self, nid: str) -> set[str]:
        return set(self.nodes[nid].pred)

    def successors(self, nid: str) -> set[str]:
        return set(self.nodes[nid].succ)

    def weight(self, nid: str) -> int:
        return self.nodes[nid].weight

    def node_ids(self) -> set[str]:
        return set(self.nodes.keys())

    def has_node(self, nid: str) -> bool:
        return nid in self.nodes

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    # ── plugin registration ──────────────────

    def register(self, kind: str, apply_fn: ApplyFn, revert_fn: RevertFn) -> None:
        """Register (or replace) handlers for an Op kind."""
        self._handlers[kind] = (apply_fn, revert_fn)

    # ── Op execution ─────────────────────────

    def execute(self, op: Op) -> None:
        """Apply op, append to history, discard any redo-future."""
        self._history = self._history[: self._cursor]
        apply_fn, _ = self._handlers[op.kind]
        apply_fn(self, op.forward)
        self._history.append(op)
        self._cursor += 1

    def undo(self) -> Op | None:
        """Revert the last executed op.  Returns the reverted Op or None."""
        if self._cursor == 0:
            return None
        self._cursor -= 1
        op = self._history[self._cursor]
        _, revert_fn = self._handlers[op.kind]
        revert_fn(self, op.inverse)
        return op

    def redo(self) -> Op | None:
        """Re-apply the next undone op.  Returns the re-applied Op or None."""
        if self._cursor >= len(self._history):
            return None
        op = self._history[self._cursor]
        apply_fn, _ = self._handlers[op.kind]
        apply_fn(self, op.forward)
        self._cursor += 1
        return op

    @property
    def history(self) -> list[Op]:
        return list(self._history)

    @property
    def cursor(self) -> int:
        return self._cursor
