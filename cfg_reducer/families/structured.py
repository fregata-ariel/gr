"""Internal structured statement templates (not a registered family)."""

from __future__ import annotations

from dataclasses import dataclass

from ..generator_types import CFGShape


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
