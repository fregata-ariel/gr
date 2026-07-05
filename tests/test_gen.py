import random

from cfg_reducer import GraphEngine, ReductionAlgorithm, metagraph, motif
from cfg_reducer.gen import build_cfg
from cfg_reducer.linearize import encode
from main import build_cfg as main_build_cfg


def _edge_set(engine) -> set[tuple[str, str]]:
    return {
        (src, dst)
        for src in engine.node_ids()
        for dst in engine.successors(src)
    }


def test_build_cfg_is_importable_from_cfg_reducer_gen_and_main_reexport():
    assert build_cfg is main_build_cfg


def test_build_cfg_is_seed_deterministic():
    first = build_cfg(num_nodes=10, edge_prob=0.2, seed=7)
    second = build_cfg(num_nodes=10, edge_prob=0.2, seed=7)

    assert _edge_set(first) == _edge_set(second)


def test_build_cfg_creates_requested_number_of_nodes():
    engine = build_cfg(num_nodes=9, edge_prob=0.2, seed=3)

    assert len(engine.node_ids()) == 9


def _graph_edges(engine: GraphEngine) -> list[tuple[str, str]]:
    return sorted(
        (src, dst)
        for src in sorted(engine.node_ids())
        for dst in sorted(engine.successors(src))
    )


def _build_graph(edges: list[tuple[str, str]]) -> GraphEngine:
    engine = GraphEngine()
    nodes = sorted({node for edge in edges for node in edge})

    for node in nodes:
        engine.add_node(node)

    for src, dst in edges:
        engine.add_edge(src, dst)

    return engine


def _extract_metagraph(
    edges: list[tuple[str, str]],
    entry: str | None = None,
):
    engine = _build_graph(edges)
    if entry is None:
        algorithm = ReductionAlgorithm(engine)
    else:
        algorithm = ReductionAlgorithm(engine, entry=entry)

    while algorithm.step() is not None:
        pass

    motifs = motif.extract(engine.history)
    return metagraph.build(motifs)


def _relabel_edges(
    edges: list[tuple[str, str]],
    mapping: dict[str, str],
) -> list[tuple[str, str]]:
    return [(mapping[src], mapping[dst]) for src, dst in edges]


def _seed_derived_name_mapping(
    node_ids: list[str],
    seed: int,
) -> dict[str, str]:
    shuffled = sorted(node_ids)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(sorted(node_ids), shuffled, strict=True))


def _dominators(
    engine: GraphEngine,
    entry: str,
) -> dict[str, set[str]]:
    nodes = sorted(engine.node_ids())
    dominators = {node: set(nodes) for node in nodes}
    dominators[entry] = {entry}

    changed = True
    while changed:
        changed = False

        for node in nodes:
            if node == entry:
                continue

            predecessors = sorted(engine.predecessors(node))
            if predecessors:
                new_dominators = {node} | set.intersection(
                    *(dominators[pred] for pred in predecessors)
                )
            else:
                new_dominators = {node}

            if new_dominators != dominators[node]:
                dominators[node] = new_dominators
                changed = True

    return dominators


def _reaches(
    engine: GraphEngine,
    src: str,
    dst: str,
) -> bool:
    stack = [src]
    seen: set[str] = set()

    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(sorted(engine.successors(node) - seen))

    return False


def _is_acyclic(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    indegree = {node: 0 for node in nodes}
    succs: dict[str, list[str]] = {node: [] for node in nodes}
    for src, dst in edges:
        succs[src].append(dst)
        indegree[dst] += 1

    frontier = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for child in succs[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)

    return visited == len(nodes)


def test_build_cfg_generates_reducible_graphs():
    violations: list[tuple[int, float, int]] = []

    for num_nodes in (12, 20):
        for edge_prob in (0.2, 0.5):
            for seed in range(20):
                engine = build_cfg(
                    num_nodes=num_nodes,
                    edge_prob=edge_prob,
                    seed=seed,
                )
                dominators = _dominators(engine, "N00")

                # Reducibility: removing every edge whose target dominates
                # its source (the back edges) must leave an acyclic graph.
                forward_edges = [
                    (src, dst)
                    for src, dst in _graph_edges(engine)
                    if dst not in dominators[src]
                ]
                if not _is_acyclic(sorted(engine.node_ids()), forward_edges):
                    violations.append((num_nodes, edge_prob, seed))

    assert not violations, (
        "graph must be acyclic after removing dominator-targeting "
        f"back edges; violations: {violations}"
    )


def test_build_cfg_still_generates_loops():
    acyclic_graphs: list[tuple[int, float, int]] = []

    for num_nodes in (12, 20):
        for edge_prob in (0.2, 0.5):
            for seed in range(20):
                engine = build_cfg(
                    num_nodes=num_nodes,
                    edge_prob=edge_prob,
                    seed=seed,
                )
                has_cycle = any(
                    _reaches(engine, dst, src)
                    for src, dst in _graph_edges(engine)
                )
                if not has_cycle:
                    acyclic_graphs.append((num_nodes, edge_prob, seed))

    assert not acyclic_graphs, (
        "reducibility must not be satisfied by generating no loops; "
        f"graphs without any cycle: {acyclic_graphs}"
    )


def test_build_cfg_is_canonical_under_seed_derived_node_id_permutations():
    mismatches: list[tuple[int, int]] = []

    for num_nodes in (12, 20):
        for seed in range(20):
            engine = build_cfg(
                num_nodes=num_nodes,
                edge_prob=0.2,
                seed=seed,
            )
            edges = _graph_edges(engine)
            mapping = _seed_derived_name_mapping(
                sorted(engine.node_ids()),
                seed,
            )
            original = encode(_extract_metagraph(edges, entry="N00"))
            relabeled = encode(
                _extract_metagraph(
                    _relabel_edges(edges, mapping),
                    entry=mapping["N00"],
                )
            )

            if original != relabeled:
                mismatches.append((num_nodes, seed))

    assert not mismatches, (
        "canonical encode output must be permutation-invariant; "
        f"mismatches: {mismatches}"
    )


def test_build_cfg_is_seed_deterministic_at_high_edge_density():
    first = build_cfg(num_nodes=20, edge_prob=0.5, seed=11)
    second = build_cfg(num_nodes=20, edge_prob=0.5, seed=11)

    assert _edge_set(first) == _edge_set(second)
