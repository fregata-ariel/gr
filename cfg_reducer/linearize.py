from __future__ import annotations

from dataclasses import dataclass

from .types import MetaGraph, Skeleton

_ADD_TOKENS = {
    "entry": "ADD_ENTRY",
    "linear": "ADD_LINEAR",
    "merge": "ADD_MERGE",
    "loop": "ADD_LOOP",
}

_TOKEN_KINDS = {token: kind for kind, token in _ADD_TOKENS.items()}

_KIND_RANK = {
    "entry": 0,
    "linear": 1,
    "merge": 2,
    "loop": 3,
}


def canonical_order(mg: MetaGraph) -> list[int]:
    motifs_by_step = {motif.step: motif for motif in mg.motifs}
    motif_positions = {
        motif.step: index
        for index, motif in enumerate(mg.motifs)
    }
    parents, children = _adjacency(mg)
    indegree = {motif.step: len(parents[motif.step]) for motif in mg.motifs}
    remaining = set(indegree)
    current_layer = sorted(step for step, degree in indegree.items() if degree == 0)
    order: list[int] = []
    canonical_index: dict[int, int] = {}

    while current_layer:
        current_layer.sort(
            key=lambda step: (
                tuple(sorted(canonical_index[parent] for parent in parents[step])),
                _KIND_RANK[motifs_by_step[step].kind],
                motif_positions[step],
            )
        )

        next_layer: list[int] = []
        for step in current_layer:
            canonical_index[step] = len(order)
            order.append(step)
            remaining.discard(step)
            for child in children[step]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_layer.append(child)

        current_layer = next_layer

    if remaining:
        raise ValueError("metagraph is not a DAG")

    return order


def encode(mg: MetaGraph) -> list[str]:
    order = canonical_order(mg)
    parents, _ = _adjacency(mg)
    canonical_index = {step: index for index, step in enumerate(order)}
    motifs_by_step = {motif.step: motif for motif in mg.motifs}
    tokens: list[str] = []

    for step in order:
        motif = motifs_by_step[step]
        tokens.append(_ADD_TOKENS[motif.kind])
        for parent in sorted(canonical_index[parent] for parent in parents[step]):
            tokens.append(f"ptr_{parent}")

        if motif.kind == "loop":
            tokens.append("OPEN")
            tokens.extend(encode(mg.subgraphs.get(step, MetaGraph((), ()))))
            tokens.append("CLOSE")

    tokens.append("STOP")
    return tokens


def skeleton_of(mg: MetaGraph) -> Skeleton:
    order = canonical_order(mg)
    parents, _ = _adjacency(mg)
    canonical_index = {step: index for index, step in enumerate(order)}
    motifs_by_step = {motif.step: motif for motif in mg.motifs}
    items: list[tuple[str, tuple[int, ...]]] = []
    subgraphs: dict[int, Skeleton] = {}

    for index, step in enumerate(order):
        motif = motifs_by_step[step]
        parent_indices = tuple(sorted(canonical_index[parent] for parent in parents[step]))
        items.append((motif.kind, parent_indices))
        if motif.kind == "loop":
            subgraphs[index] = skeleton_of(mg.subgraphs.get(step, MetaGraph((), ())))

    return Skeleton(items=tuple(items), subgraphs=subgraphs)


def decode(tokens: list[str]) -> Skeleton:
    parser = _Parser(tokens)
    skeleton = parser.parse_level(expect_close=False)
    if parser.pos != len(tokens):
        parser.error("trailing tokens after top-level STOP", parser.pos)
    return skeleton


def _adjacency(
    mg: MetaGraph,
) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    steps = [motif.step for motif in mg.motifs]
    parents = {step: set() for step in steps}
    children = {step: [] for step in steps}

    for src, dst in set(mg.edges):
        if src in children and dst in parents and src != dst:
            parents[dst].add(src)
            children[src].append(dst)

    for child_steps in children.values():
        child_steps.sort()

    return parents, children


@dataclass
class _Parser:
    tokens: list[str]
    pos: int = 0

    def parse_level(self, expect_close: bool) -> Skeleton:
        items: list[tuple[str, tuple[int, ...]]] = []
        subgraphs: dict[int, Skeleton] = {}

        while True:
            if self.pos >= len(self.tokens):
                if expect_close:
                    self.error("stream ended before CLOSE", self.pos)
                self.error("stream ended without top-level STOP", self.pos)

            token = self.tokens[self.pos]

            if token == "STOP":
                self.pos += 1
                break

            if token == "CLOSE":
                if expect_close:
                    self.error("CLOSE without preceding child-level STOP", self.pos)
                self.error("unexpected CLOSE at top level", self.pos)

            kind = _TOKEN_KINDS.get(token)
            if kind is None:
                self.error(f"unknown token {token!r}", self.pos)
            assert kind is not None

            self.pos += 1
            parents = self.parse_parents(len(items))
            current_index = len(items)
            items.append((kind, parents))

            if kind == "loop":
                if self.pos >= len(self.tokens) or self.tokens[self.pos] != "OPEN":
                    self.error("OPEN must follow ADD_LOOP parents", self.pos)
                self.pos += 1
                subgraphs[current_index] = self.parse_level(expect_close=True)
                if self.pos >= len(self.tokens) or self.tokens[self.pos] != "CLOSE":
                    self.error("unbalanced OPEN/CLOSE", self.pos)
                self.pos += 1
            elif self.pos < len(self.tokens) and self.tokens[self.pos] == "OPEN":
                self.error("OPEN must follow ADD_LOOP parents", self.pos)

        return Skeleton(items=tuple(items), subgraphs=subgraphs)

    def parse_parents(self, n_items: int) -> tuple[int, ...]:
        parents: list[int] = []
        previous = -1

        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if not token.startswith("ptr_"):
                break

            try:
                parent = int(token[4:])
            except ValueError as exc:
                raise ValueError(
                    f"invalid pointer token at position {self.pos}: {token!r}"
                ) from exc

            if parent >= n_items:
                self.error("forward/self ptr reference", self.pos)
            if parent <= previous:
                if parent == previous:
                    self.error("duplicate ptr in parents list", self.pos)
                self.error("non-ascending ptrs in parents list", self.pos)

            parents.append(parent)
            previous = parent
            self.pos += 1

        return tuple(parents)

    def error(self, message: str, pos: int) -> None:
        raise ValueError(f"{message} at position {pos}")
