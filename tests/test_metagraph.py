from cfg_reducer import GraphEngine, MetaGraph, ReductionAlgorithm, motif, metagraph


def _build_graph(edges: list[tuple[str, str]]) -> GraphEngine:
    engine = GraphEngine()
    nodes = sorted({n for edge in edges for n in edge})

    for node in nodes:
        engine.add_node(node)

    for src, dst in edges:
        engine.add_edge(src, dst)

    return engine


def _extract_metagraph(edges: list[tuple[str, str]]) -> MetaGraph:
    engine = _build_graph(edges)
    algorithm = ReductionAlgorithm(engine)

    while algorithm.step() is not None:
        pass

    motifs = motif.extract(engine.history)
    return metagraph.build(motifs)


def _node_steps(mg: MetaGraph) -> dict[str, int]:
    return {
        m.node: m.step
        for m in mg.motifs
        if m.node is not None
    }


def test_diamond_metagraph():
    mg = _extract_metagraph([
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
    ])
    steps = _node_steps(mg)

    assert len(mg.motifs) == 4
    assert len(mg.edges) == 4
    assert mg.subgraphs == {}
    assert set(mg.edges) == {
        (steps["A"], steps["B"]),
        (steps["A"], steps["C"]),
        (steps["B"], steps["D"]),
        (steps["C"], steps["D"]),
    }


def test_simple_loop_metagraph():
    mg = _extract_metagraph([
        ("A", "B"),
        ("B", "C"),
        ("C", "B"),
        ("C", "D"),
    ])
    steps = _node_steps(mg)
    loop = next(m for m in mg.motifs if m.kind == "loop")

    assert len(mg.motifs) == 3
    assert len(mg.edges) == 2
    assert set(mg.edges) == {
        (steps["A"], loop.step),
        (loop.step, steps["D"]),
    }
    assert len(mg.subgraphs) == 1

    sub = mg.subgraphs[loop.step]
    assert len(sub.motifs) == 2
    assert len(sub.edges) == 1
    assert sub.subgraphs == {}


def test_nested_loop_metagraph():
    mg = _extract_metagraph([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "C"),
        ("D", "B"),
        ("B", "E"),
    ])
    steps = _node_steps(mg)
    outer_loop = next(m for m in mg.motifs if m.kind == "loop")

    assert len(mg.motifs) == 3
    assert len(mg.edges) == 2
    assert set(mg.edges) == {
        (steps["A"], outer_loop.step),
        (outer_loop.step, steps["E"]),
    }
    assert len(mg.subgraphs) == 1

    outer_sub = mg.subgraphs[outer_loop.step]
    inner_loop = next(m for m in outer_sub.motifs if m.kind == "loop")
    outer_steps = _node_steps(outer_sub)

    assert len(outer_sub.subgraphs) == 1
    assert set(outer_sub.edges) == {(outer_steps["B"], inner_loop.step)}

    inner_sub = outer_sub.subgraphs[inner_loop.step]
    assert len(inner_sub.motifs) == 2
    assert len(inner_sub.edges) == 1
    assert inner_sub.subgraphs == {}
