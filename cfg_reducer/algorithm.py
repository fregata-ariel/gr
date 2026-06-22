"""
Reverse-Kahn reduction algorithm with cycle breaking.

Reads graph state via engine queries, emits Ops for engine to execute.
The algorithm itself never mutates the graph directly.

The two-phase loop:
    Phase 1  (reverse Kahn)  — pop smallest-weight terminal, propagate
    Phase 2  (cycle break)   — find terminal SCC, cut back-edges to header
    repeat until graph is empty.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field

from .types import Op
from .engine import GraphEngine


# ──────────────────────────────────────────────
#  Scope — zoom into a sub-region (SCC)
# ──────────────────────────────────────────────

@dataclass
class Scope:
    active: bool = False
    focus: set[str] = field(default_factory=set)

    def enter(self, nodes: set[str]) -> None:
        self.active = True
        self.focus = set(nodes)

    def leave(self) -> None:
        self.active = False
        self.focus = set()

    def contains(self, nid: str) -> bool:
        """Node is in scope?  (When inactive, everything is in scope.)"""
        return (not self.active) or (nid in self.focus)

    def snapshot(self) -> dict:
        return {"active": self.active, "focus": set(self.focus)}

    def restore(self, snap: dict) -> None:
        self.active = snap["active"]
        self.focus = set(snap["focus"])


# ──────────────────────────────────────────────
#  Tarjan SCC  (iterative, no recursion limit)
# ──────────────────────────────────────────────

def tarjan_scc(node_ids: set[str], succ_fn) -> list[set[str]]:
    """
    Find all strongly connected components.

    Args:
        node_ids  — set of node ids to consider
        succ_fn   — succ_fn(nid) -> iterable of successor ids
                     (only those within node_ids are followed)

    Returns:
        list of sets, each set is one SCC.
        Ordered in reverse-finishing order (leaves first).
    """
    index_counter = 0
    index_map: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[set[str]] = []

    # Iterative DFS with explicit call stack
    # Frame: (node, successors_iterator, phase)
    #   phase 0 = first visit,  phase 1 = returning from child
    call_stack: list[tuple[str, list[str], int, int]] = []
    #                   node   children   child_idx  phase

    for start in sorted(node_ids):      # sorted for determinism
        if start in index_map:
            continue

        call_stack.append((start, [], 0, 0))

        while call_stack:
            node, children, ci, phase = call_stack.pop()

            if phase == 0:
                # First visit
                index_map[node] = index_counter
                lowlink[node] = index_counter
                index_counter += 1
                on_stack.add(node)
                stack.append(node)

                # Collect children (only within node_ids)
                children = [
                    s for s in succ_fn(node)
                    if s in node_ids
                ]
                ci = 0

            # Process children from where we left off
            while ci < len(children):
                child = children[ci]
                if child not in index_map:
                    # Push current frame (will resume at phase 1)
                    call_stack.append((node, children, ci, 1))
                    # Push child as new frame
                    call_stack.append((child, [], 0, 0))
                    break
                elif child in on_stack:
                    lowlink[node] = min(lowlink[node], index_map[child])
                ci += 1
            else:
                # All children processed — check if root of SCC
                if lowlink[node] == index_map[node]:
                    scc: set[str] = set()
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        scc.add(w)
                        if w == node:
                            break
                    result.append(scc)

                # Propagate lowlink to parent
                if call_stack:
                    parent_node = call_stack[-1][0]
                    lowlink[parent_node] = min(
                        lowlink[parent_node], lowlink[node]
                    )
                    # Advance parent's child index
                    pn, pc, pci, pp = call_stack[-1]
                    call_stack[-1] = (pn, pc, pci + 1, pp)

    return result


# ──────────────────────────────────────────────
#  ReductionAlgorithm
# ──────────────────────────────────────────────

class ReductionAlgorithm:
    """
    Drives the two-phase reverse-Kahn reduction.
    Call step() repeatedly; it returns the Op applied, or None when done.
    """

    def __init__(self, engine: GraphEngine) -> None:
        self.engine = engine
        self.scope = Scope()
        self.heap: list[tuple[int, str]] = []   # (weight, node_id)

        # Seed heap with initial terminals
        self._collect_terminals(engine.node_ids())

    # ── public interface ─────────────────────

    def step(self) -> Op | None:
        """Execute one reduction step.  Returns the Op, or None if done."""
        # Drain stale entries
        self._purge_heap()

        if self.heap:
            return self._reduce_terminal()

        if not self.engine.is_empty():
            return self._break_cycle()

        return None     # fully reduced

    def undo(self) -> Op | None:
        """Undo one step.  Restores graph and algorithm state."""
        op = self.engine.undo()
        if op is None:
            return None

        # Restore scope from snapshot stored in op.meta
        if "scope_snapshot" in op.meta:
            self.scope.restore(op.meta["scope_snapshot"])

        # Rebuild heap from current graph + scope
        self._rebuild_heap()
        return op

    def redo(self) -> Op | None:
        """Redo one undone step."""
        op = self.engine.redo()
        if op is None:
            return None

        if "scope_after" in op.meta:
            self.scope.restore(op.meta["scope_after"])

        self._rebuild_heap()
        return op

    @property
    def is_done(self) -> bool:
        self._purge_heap()
        return len(self.heap) == 0 and self.engine.is_empty()

    # ── Phase 1: terminal reduction ──────────

    def _reduce_terminal(self) -> Op:
        _, u = heapq.heappop(self.heap)

        node = self.engine.nodes[u]
        preds = sorted(node.pred)
        succs = sorted(node.succ)
        weight = node.weight
        ntype = node.node_type

        # Snapshot scope BEFORE mutation
        scope_before = self.scope.snapshot()

        # Build inverse metadata
        weight_deltas = {
            p: weight for p in preds if self.engine.has_node(p)
        }

        op = Op(
            kind="remove_node",
            forward={"target": u},
            inverse={
                "target": u,
                "pred_edges": preds,
                "succ_edges": succs,
                "weight": weight,
                "node_type": ntype,
                "weight_deltas": weight_deltas,
            },
        )

        self.engine.execute(op)

        # Update scope
        if self.scope.active:
            self.scope.focus.discard(u)

        # Enqueue new terminals among predecessors
        for p in preds:
            if self.engine.has_node(p) and self.engine.out_degree(p) == 0:
                if self.scope.contains(p):
                    heapq.heappush(
                        self.heap, (self.engine.weight(p), p)
                    )

        # If scope is exhausted, zoom out and re-collect
        scope_after = self.scope.snapshot()
        if self.scope.active and not self.scope.focus:
            self.scope.leave()
            scope_after = self.scope.snapshot()
            self._collect_terminals(self.engine.node_ids())

        op.meta["scope_snapshot"] = scope_before
        op.meta["scope_after"] = scope_after
        return op

    # ── Phase 2: cycle breaking ──────────────

    def _break_cycle(self) -> Op:
        scope_before = self.scope.snapshot()

        search_nodes = (
            self.scope.focus if self.scope.active
            else self.engine.node_ids()
        )

        # Find SCCs within search scope
        sccs = tarjan_scc(
            search_nodes,
            lambda nid: self.engine.successors(nid),
        )

        # Pick a terminal SCC (no out-edges to other nodes in scope)
        target_scc = self._pick_terminal_scc(sccs, search_nodes)

        # Identify header: node with external (outside SCC) predecessors
        header = self._identify_header(target_scc)

        # Collect back-edges:  (u -> header) where u ∈ SCC
        back_edges = sorted(
            (u, header)
            for u in target_scc
            if header in self.engine.successors(u)
        )

        op = Op(
            kind="remove_edges",
            forward={"edges": back_edges},
            inverse={"edges": back_edges},
        )

        self.engine.execute(op)

        # Enter scope on this SCC
        self.scope.enter(target_scc)

        # Enqueue new terminals within the SCC
        self.heap = []
        self._collect_terminals(self.scope.focus)

        scope_after = self.scope.snapshot()
        op.meta["scope_snapshot"] = scope_before
        op.meta["scope_after"] = scope_after
        op.meta["header"] = header
        op.meta["scc"] = sorted(target_scc)
        return op

    # ── helpers ───────────────────────────────

    def _pick_terminal_scc(
        self, sccs: list[set[str]], scope: set[str]
    ) -> set[str]:
        """
        Choose an SCC with no out-edges leaving it (within scope).
        Among candidates, pick the smallest for determinism.
        """
        for scc in sccs:
            if len(scc) < 2:
                continue
            has_exit = any(
                s not in scc
                for n in scc
                for s in self.engine.successors(n)
                if s in scope
            )
            if not has_exit:
                return scc

        # Fallback: largest non-trivial SCC (should not normally reach here)
        non_trivial = [s for s in sccs if len(s) >= 2]
        return max(non_trivial, key=len)

    def _identify_header(self, scc: set[str]) -> str:
        """
        Header = node reachable from outside the SCC.
        Falls back to min-id for isolated loops.
        """
        external_entries: list[str] = []
        for n in scc:
            for p in self.engine.predecessors(n):
                if p not in scc:
                    external_entries.append(n)
                    break

        if external_entries:
            return min(external_entries)
        return min(scc)

    def _collect_terminals(self, candidates: set[str]) -> None:
        """Push all out_degree==0 nodes from candidates into the heap."""
        for nid in candidates:
            if self.engine.has_node(nid) and self.engine.out_degree(nid) == 0:
                heapq.heappush(
                    self.heap, (self.engine.weight(nid), nid)
                )

    def _rebuild_heap(self) -> None:
        """Reconstruct the heap from scratch (used after undo/redo)."""
        self.heap = []
        scope_nodes = (
            self.scope.focus if self.scope.active
            else self.engine.node_ids()
        )
        self._collect_terminals(scope_nodes)

    def _purge_heap(self) -> None:
        """Drop entries for nodes already removed from the graph."""
        while self.heap and not self.engine.has_node(self.heap[0][1]):
            heapq.heappop(self.heap)
