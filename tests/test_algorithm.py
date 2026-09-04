"""ReductionAlgorithm and tarjan_scc regression tests."""

from cfg_reducer import GraphEngine, ReductionAlgorithm, tarjan_scc

DIAMOND = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
SIMPLE_LOOP = [("A", "B"), ("B", "C"), ("C", "B"), ("C", "D")]
NESTED_LOOP = [
    ("A", "B"), ("B", "C"), ("C", "D"),
    ("D", "C"), ("D", "B"), ("B", "E"),
]


def _build(edges: list[tuple[str, str]]) -> GraphEngine:
    engine = GraphEngine()
    for n in sorted({n for e in edges for n in e}):
        engine.add_node(n)
    for src, dst in edges:
        engine.add_edge(src, dst)
    return engine


def _succ_fn(edges: list[tuple[str, str]]):
    table: dict[str, set[str]] = {}
    for src, dst in edges:
        table.setdefault(src, set()).add(dst)
    return lambda nid: table.get(nid, set())


def _run_to_completion(engine: GraphEngine) -> ReductionAlgorithm:
    algorithm = ReductionAlgorithm(engine)
    while algorithm.step() is not None:
        pass
    return algorithm


def _snapshot(engine: GraphEngine) -> dict:
    return {
        nid: (set(n.pred), set(n.succ), n.weight, n.node_type)
        for nid, n in engine.nodes.items()
    }


# ── tarjan_scc ───────────────────────────────

def test_tarjan_dag_yields_singletons():
    nodes = {n for e in DIAMOND for n in e}
    sccs = tarjan_scc(nodes, _succ_fn(DIAMOND))

    assert all(len(scc) == 1 for scc in sccs)
    assert set().union(*sccs) == nodes


def test_tarjan_finds_single_cycle():
    nodes = {n for e in SIMPLE_LOOP for n in e}
    sccs = tarjan_scc(nodes, _succ_fn(SIMPLE_LOOP))

    assert {frozenset(s) for s in sccs} == {
        frozenset({"B", "C"}), frozenset({"A"}), frozenset({"D"}),
    }


def test_tarjan_multiple_sccs_leaves_first():
    edges = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "D"), ("D", "C")]
    nodes = {n for e in edges for n in e}
    sccs = tarjan_scc(nodes, _succ_fn(edges))

    non_trivial = [frozenset(s) for s in sccs if len(s) >= 2]
    assert non_trivial == [frozenset({"C", "D"}), frozenset({"A", "B"})]


def test_tarjan_is_deterministic():
    nodes = {n for e in NESTED_LOOP for n in e}
    first = tarjan_scc(nodes, _succ_fn(NESTED_LOOP))
    second = tarjan_scc(nodes, _succ_fn(NESTED_LOOP))
    assert first == second


# ── ReductionAlgorithm ───────────────────────

def test_dag_reduces_to_empty_without_cycle_breaking():
    engine = _build(DIAMOND)
    algorithm = _run_to_completion(engine)

    assert algorithm.is_done
    assert engine.is_empty()
    assert [op.kind for op in engine.history] == ["remove_node"] * 4


def test_loop_reduction_records_scc_metadata():
    engine = _build(SIMPLE_LOOP)
    _run_to_completion(engine)

    cuts = [op for op in engine.history if op.kind == "remove_edges"]
    assert len(cuts) == 1
    assert cuts[0].meta["scc"] == ["B", "C"]
    assert cuts[0].meta["header"] == "B"
    assert cuts[0].forward["edges"] == [("C", "B")]


def test_full_undo_restores_initial_graph():
    engine = _build(NESTED_LOOP)
    initial = _snapshot(engine)

    algorithm = _run_to_completion(engine)
    assert engine.is_empty()

    while algorithm.undo() is not None:
        pass
    assert _snapshot(engine) == initial
    assert engine.cursor == 0

    # Redo must reach the fully reduced state again
    while algorithm.redo() is not None:
        pass
    assert engine.is_empty()


# ── determinism: independent of successor iteration order ──

def test_tarjan_scc_order_is_independent_of_successor_iteration_order():
    from cfg_reducer.algorithm import tarjan_scc
    # entry E feeds two sibling loops L1<->L1a and L2<->L2a
    adj = {"E": ["L1", "L2"], "L1": ["L1a"], "L1a": ["L1"],
           "L2": ["L2a"], "L2a": ["L2"]}
    nodes = set(adj)
    forward = tarjan_scc(nodes, lambda n: adj[n])
    reverse = tarjan_scc(nodes, lambda n: list(reversed(adj[n])))
    assert forward == reverse


def test_reduction_sketch_is_independent_of_successor_iteration_order():
    """Seed 300607 / 24 nodes flipped its sibling-loop order between
    processes before children were sorted in tarjan_scc."""
    from cfg_reducer import dataset, model_input
    from cfg_reducer.generate import generate_cfg

    def sketch(order):
        engine = GraphEngine()
        generate_cfg(engine, num_nodes=24, edge_prob=0.18, seed=300607)
        original = engine.successors
        engine.successors = lambda nid: sorted(original(nid), reverse=order)  # ty: ignore[invalid-assignment]
        return model_input.sketch_of(dataset.reduce_to_metagraph(engine))

    assert sketch(False) == sketch(True)
