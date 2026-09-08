"""Structured CFG planning, template lowering, and bounded generation."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from random import Random
from typing import cast

from ..generator_types import CFGShape, GeneratorSpec, Json


@dataclass(frozen=True)
class Stmt:
    kind: str
    children: tuple[Stmt, ...] = ()

    def __post_init__(self) -> None:
        arities = {"atom": 0, "if": 1, "if_else": 2, "while": 1,
                   "do_while": 1, "break": 0, "continue": 0, "return": 0}
        if not isinstance(self.children, tuple) or not all(
            isinstance(child, Stmt) for child in self.children
        ):
            raise ValueError("children must be a tuple of Stmt")
        if self.kind == "seq":
            return
        if self.kind == "switch":
            if len(self.children) >= 3:
                return
        elif self.kind in arities and len(self.children) == arities[self.kind]:
            return
        raise ValueError(f"invalid statement kind or child count: {self.kind}")


def lower_structure(tree: Stmt, merge_degree: int) -> CFGShape:
    """Reserve templates in preorder, then route joins and assign public IDs.

    Abrupt leaves have no normal exit and must be guarded by an if to
    preserve a normal path. In seq(if(abrupt), atom), the if's join is the
    original atom, so the conditional insertion costs exactly two nodes.
    """
    if type(merge_degree) is not int or merge_degree < 2:
        raise ValueError("merge_degree must be an integer >= 2")
    # ENTRY and EXIT are reserved first; EXIT is moved last only at naming.
    incoming: list[list[int]] = [[], []]
    contexts: list[tuple[int, int, int]] = []

    def node() -> int:
        incoming.append([])
        return len(incoming) - 1

    def connect(sources: list[int], dst: int) -> None:
        incoming[dst].extend(sources)

    def lower(stmt: Stmt) -> tuple[int, list[int]]:
        kind = stmt.kind
        if kind == "atom" or (kind == "seq" and not stmt.children):
            atom = node()
            return atom, [atom]
        if kind == "seq":
            first: int | None = None
            exits: list[int] = []
            previous: Stmt | None = None
            for child in stmt.children:
                # Share the dedicated join with the following original atom.
                if (child.kind == "atom" and previous is not None
                        and previous.kind == "if"
                        and previous.children[0].kind in ("break", "continue", "return")):
                    previous = child
                    continue
                start, ends = lower(child)
                if first is None:
                    first = start
                else:
                    connect(exits, start)
                exits = ends
                previous = child
            assert first is not None
            return first, exits
        if kind in ("break", "continue", "return"):
            if kind != "return" and not contexts:
                raise ValueError(f"{kind} requires a loop context")
            abrupt = node()
            target = 1 if kind == "return" else contexts[-1][2 if kind == "break" else 1]
            connect([abrupt], target)
            return abrupt, []
        if kind in ("while", "do_while"):
            header, tail, end = node(), node(), node()
            contexts.append((header, tail, end))
            start, exits = lower(stmt.children[0])
            contexts.pop()
            connect([header], start)
            connect(exits, tail)
            connect([tail], header)
            connect([header if kind == "while" else tail], end)
            return header, [end]
        condition, join = node(), node()
        for child in stmt.children:
            start, exits = lower(child)
            connect([condition], start)
            connect(exits, join)
        if kind == "if":
            connect([condition], join)
        return condition, [join]

    start, exits = lower(tree)
    connect([0], start)
    connect(exits, 1)
    edges: list[tuple[int, int]] = []
    # Route every reserved destination, including EXIT, in creation order.
    for dst in range(len(incoming)):
        sources = sorted(set(incoming[dst]))
        while len(sources) > merge_degree:
            relay = node()
            edges.extend((src, relay) for src in sources[:merge_degree])
            sources = [relay] + sources[merge_degree:]
        edges.extend((src, dst) for src in sources)
    order = [0, *range(2, len(incoming)), 1]
    names = {symbol: f"N{i:02d}" for i, symbol in enumerate(order)}
    return CFGShape(tuple(names[symbol] for symbol in order),
                    tuple(sorted((names[u], names[v]) for u, v in edges)), names[0])


class GenerationRejected(Exception):
    """A known, seed-dependent generation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_DEFAULTS: dict[str, Json] = {
    "max_layer_width": 3, "branch_degree": 2, "merge_degree": 3,
    "loop_count": 2, "goto_count": 1, "span_mode": "uniform",
    "target_depth": None,
}
_LOOPS = ("while", "do_while")


def _atoms(tree: Stmt, path: tuple[int, ...] = (), depth: int = 0
           ) -> list[tuple[tuple[int, ...], int]]:
    """Return occurrence paths (not statement identities) in preorder."""
    if tree.kind == "atom":
        return [(path, depth)]
    return [item for i, child in enumerate(tree.children)
            for item in _atoms(child, (*path, i), depth + (tree.kind in _LOOPS))]


def _replace(tree: Stmt, path: tuple[int, ...], replacement: Stmt) -> Stmt:
    if not path:
        return replacement
    i, *rest = path
    children = list(tree.children)
    children[i] = _replace(children[i], tuple(rest), replacement)
    return Stmt(tree.kind, tuple(children))


def plan_structure(spec: GeneratorSpec, rng: Random) -> Stmt:
    spec = Structured().normalize(spec)
    loops = cast(int, spec.params["loop_count"])
    depth = cast(int | None, spec.params["target_depth"])
    if depth is None:
        depth = rng.randint(1, min(loops, 3)) if loops else 0
    tree = Stmt("atom")
    for _ in range(depth):
        tree = Stmt(rng.choice(_LOOPS), (tree,))
    tree = Stmt("seq", (tree, *(Stmt(rng.choice(_LOOPS), (Stmt("atom"),))
                               for _ in range(loops - depth))))
    for _ in range(cast(int, spec.params["goto_count"])):
        path, context = rng.choice(_atoms(tree))
        abrupt = Stmt("if", (Stmt(rng.choice(
            ("break", "continue", "return") if context else ("return",))),))
        # Splice into an existing seq; otherwise lift the lone atom into one.
        parent = tree
        for i in path[:-1]:
            parent = parent.children[i]
        if path and parent.kind == "seq":
            i = path[-1]
            replacement = Stmt("seq", (*parent.children[:i], abrupt,
                                       *parent.children[i:]))
            tree = _replace(tree, path[:-1], replacement)
        else:
            tree = _replace(tree, path, Stmt("seq", (abrupt, Stmt("atom"))))
    return tree


def pad_shape(shape: CFGShape, target: int, mode: str, rng: Random) -> CFGShape:
    """Split forward edges, biasing how many existing spans are lengthened."""
    if type(target) is not int or target < len(shape.nodes):
        raise ValueError("target must be an integer >= the current node count")
    if mode not in ("short", "long", "uniform"):
        raise ValueError("span_mode must be uniform, short, or long")
    # Integer symbols avoid collisions with arbitrary input node names.
    symbols = {name: i for i, name in enumerate(shape.nodes)}
    nodes = list(range(len(shape.nodes)))
    edges = [(symbols[u], symbols[v]) for u, v in shape.edges]
    while len(nodes) < target:
        positions = {node: i for i, node in enumerate(nodes)}
        degree = Counter(u for u, _ in edges)
        candidates = sorted(((u, v) for u, v in edges if positions[u] < positions[v]),
                            key=lambda edge: (f"N{positions[edge[0]]:02d}",
                                              f"N{positions[edge[1]]:02d}"))
        spans = [(positions[u], positions[v]) for u, v in candidates
                 if degree[u] >= 2 or positions[v] - positions[u] >= 2]
        if not candidates:
            candidates = sorted(edge for edge in edges if edge[0] == symbols[shape.entry])
        if not candidates:
            raise ValueError("padding requires an ENTRY edge")
        if mode != "uniform":
            scores = [sum(lo < positions[v] < hi for lo, hi in spans)
                      for _, v in candidates]
            best = min(scores) if mode == "short" else max(scores)
            candidates = [edge for edge, score in zip(candidates, scores) if score == best]
        src, dst = rng.choice(candidates)
        new = len(nodes)
        nodes.insert(positions[dst], new)
        edges.remove((src, dst))
        edges.extend(((src, new), (new, dst)))
    names = {node: f"N{i:02d}" for i, node in enumerate(nodes)}
    return CFGShape(tuple(names[node] for node in nodes),
                    tuple(sorted((names[u], names[v]) for u, v in edges)),
                    names[symbols[shape.entry]])


class Structured:
    name = "structured"

    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec:
        spec = GeneratorSpec(spec.family, spec.num_nodes, spec.params)
        if spec.family != self.name:
            raise ValueError("structured requires family='structured'")
        if spec.num_nodes < 3:
            raise ValueError("structured requires num_nodes >= 3")
        unknown = set(spec.params) - _DEFAULTS.keys()
        if unknown:
            raise ValueError(f"unknown structured params: {sorted(unknown)}")
        params = _DEFAULTS | spec.params
        for key, minimum in (("max_layer_width", 1), ("branch_degree", 2),
                             ("merge_degree", 2), ("loop_count", 0), ("goto_count", 0)):
            value = params[key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        loops, gotos = cast(int, params["loop_count"]), cast(int, params["goto_count"])
        depth = params["target_depth"]
        if depth is not None and (type(depth) is not int or
                                  not (depth == 0 if loops == 0 else 1 <= depth <= loops)):
            raise ValueError("target_depth must be 0 without loops, otherwise 1..loop_count")
        if params["span_mode"] not in ("uniform", "short", "long"):
            raise ValueError("span_mode must be uniform, short, or long")
        if params["max_layer_width"] == 1 and (loops or gotos):
            raise ValueError("width=1 cannot request control statements")
        if 3 + 3 * loops + 2 * gotos > 11 * spec.num_nodes // 10:
            raise ValueError("node_budget: impossible minimum structure size")
        return GeneratorSpec(self.name, spec.num_nodes, params)

    def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
        spec = self.normalize(spec)
        lo, hi = (9 * spec.num_nodes + 9) // 10, 11 * spec.num_nodes // 10
        budget = rng.randint(lo, hi)
        tree = plan_structure(spec, rng)
        merge = cast(int, spec.params["merge_degree"])
        shape = lower_structure(tree, merge)
        if len(shape.nodes) > hi:
            raise GenerationRejected("node_budget")
        budget = max(budget, len(shape.nodes))
        count = rng.randint(0, (budget - len(shape.nodes)) // 2)
        width = min(cast(int, spec.params["max_layer_width"]),
                    cast(int, spec.params["branch_degree"]))
        for _ in range(count):
            candidates: list[tuple[Stmt, CFGShape]] = []
            for path, _depth in _atoms(tree):
                variants = ([Stmt("if", (Stmt("atom"),)),
                             Stmt("if_else", (Stmt("atom"),) * 2)] if width >= 2 else [])
                variants.extend(Stmt("switch", (Stmt("atom"),) * k)
                                for k in range(3, width + 1))
                for variant in variants:
                    expanded = _replace(tree, path, variant)
                    lowered = lower_structure(expanded, merge)
                    if len(lowered.nodes) <= budget:
                        candidates.append((expanded, lowered))
            if not candidates:
                break
            tree, shape = rng.choice(candidates)
        shape = pad_shape(shape, budget, cast(str, spec.params["span_mode"]), rng)
        assert len(shape.nodes) == budget
        return shape
