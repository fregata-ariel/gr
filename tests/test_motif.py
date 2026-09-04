"""Motif extraction regression tests: kinds, interfaces, hierarchy, steps."""

from cfg_reducer import GraphEngine, Motif, Op, ReductionAlgorithm, motif


def _extract(edges: list[tuple[str, str]]) -> list[Motif]:
    engine = GraphEngine()
    for n in sorted({n for e in edges for n in e}):
        engine.add_node(n)
    for src, dst in edges:
        engine.add_edge(src, dst)

    algorithm = ReductionAlgorithm(engine)
    while algorithm.step() is not None:
        pass
    return motif.extract(engine.history)


def _all_steps(motifs: list[Motif]) -> list[int]:
    steps = []
    for m in motifs:
        steps.append(m.step)
        steps.extend(_all_steps(list(m.children)))
    return steps


def _by_node(motifs: list[Motif]) -> dict[str, Motif]:
    return {m.node: m for m in motifs if m.node is not None}


def test_diamond_extracts_flat_motifs():
    motifs = _extract([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    by_node = _by_node(motifs)

    assert len(motifs) == 4
    assert all(m.children == () for m in motifs)
    assert sorted(_all_steps(motifs)) == [0, 1, 2, 3]

    assert by_node["A"].kind == "entry"
    assert by_node["A"].step == 0
    assert by_node["B"].kind == "linear"
    assert by_node["B"].preds == ("A",)
    assert by_node["C"].kind == "linear"
    assert by_node["C"].preds == ("A",)
    assert by_node["D"].kind == "merge"
    assert by_node["D"].preds == ("B", "C")
    assert by_node["D"].step == 3


def test_simple_loop_becomes_container():
    motifs = _extract([("A", "B"), ("B", "C"), ("C", "B"), ("C", "D")])

    assert [m.kind for m in motifs] == ["entry", "loop", "linear"]
    entry, loop, tail = motifs

    assert entry.node == "A" and entry.step == 0
    assert tail.node == "D" and tail.preds == ("C",)

    # Loop external interface and metadata
    assert loop.node is None
    assert loop.preds == ("A",)
    assert loop.succs == ("D",)
    assert loop.meta["header"] == "B"
    assert loop.meta["scc"] == ["B", "C"]
    assert loop.meta["back_edges"] == [("C", "B")]

    # Children carry the internal structure (classified against the
    # full graph at removal time, so the header keeps its external pred)
    children = _by_node(list(loop.children))
    assert set(children) == {"B", "C"}
    assert children["B"].kind == "linear" and children["B"].preds == ("A",)
    assert children["C"].kind == "linear" and children["C"].preds == ("B",)


def test_nested_loop_hierarchy_and_step_uniqueness():
    motifs = _extract([
        ("A", "B"), ("B", "C"), ("C", "D"),
        ("D", "C"), ("D", "B"), ("B", "E"),
    ])

    assert [m.kind for m in motifs] == ["entry", "loop", "linear"]
    outer = motifs[1]
    assert outer.preds == ("A",)
    assert outer.succs == ("E",)
    assert outer.meta["scc"] == ["B", "C", "D"]

    inner = next(m for m in outer.children if m.kind == "loop")
    assert inner.preds == ("B",)
    assert inner.succs == ()
    assert inner.meta["scc"] == ["C", "D"]
    assert inner.meta["back_edges"] == [("D", "C")]

    inner_children = _by_node(list(inner.children))
    assert set(inner_children) == {"C", "D"}

    # step is a persistent, sample-wide unique id across the hierarchy
    steps = _all_steps(motifs)
    assert sorted(steps) == list(range(7))


def test_parent_map_assigns_outer_members_removed_after_inner_scope():
    """Regression: consecutive nested cuts (scc 5 -> 3) where the outer-only
    members are removed AFTER the inner scope drains (the algorithm's Scope
    is overwritten, not stacked). They must still parent to the outer cut."""
    ops = [
        Op(kind="remove_edges", forward={"edges": [("E", "A")]},
           inverse={"edges": [("E", "A")]},
           meta={"header": "A", "scc": ["A", "B", "C", "D", "E"]}),
        Op(kind="remove_edges", forward={"edges": [("E", "C")]},
           inverse={"edges": [("E", "C")]},
           meta={"header": "C", "scc": ["C", "D", "E"]}),
        Op(kind="remove_node", forward={"target": "E"},
           inverse={"target": "E", "pred_edges": ["D"], "succ_edges": [],
                    "weight": 1, "node_type": "basic", "weight_deltas": {}}),
        Op(kind="remove_node", forward={"target": "D"},
           inverse={"target": "D", "pred_edges": ["C"], "succ_edges": [],
                    "weight": 1, "node_type": "basic", "weight_deltas": {}}),
        Op(kind="remove_node", forward={"target": "C"},
           inverse={"target": "C", "pred_edges": ["B"], "succ_edges": [],
                    "weight": 1, "node_type": "basic", "weight_deltas": {}}),
        # outer-only members, removed after the inner scope emptied
        Op(kind="remove_node", forward={"target": "B"},
           inverse={"target": "B", "pred_edges": ["A"], "succ_edges": [],
                    "weight": 1, "node_type": "basic", "weight_deltas": {}}),
        Op(kind="remove_node", forward={"target": "A"},
           inverse={"target": "A", "pred_edges": [], "succ_edges": [],
                    "weight": 1, "node_type": "basic", "weight_deltas": {}}),
    ]
    from cfg_reducer.motif import _build_parent_map
    assert _build_parent_map(ops) == {
        1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0,
    }

    motifs = motif.extract(ops)
    assert [m.kind for m in motifs] == ["loop"]
    outer = motifs[0]
    assert {m.node for m in outer.children if m.node} == {"A", "B"}
    inner = next(m for m in outer.children if m.kind == "loop")
    assert {m.node for m in inner.children} == {"C", "D", "E"}


def test_unknown_op_kind_is_preserved_not_classified():
    """A5-1: unknown Op kinds must not break extraction. They surface as
    placeholder Motifs; the canonical vocabulary stays entry/linear/merge/
    loop, and encoders must reject or skip anything else explicitly."""
    motifs = motif.extract([Op(kind="annotate")])

    assert len(motifs) == 1
    placeholder = motifs[0]
    assert placeholder.kind == "annotate"
    assert placeholder.node is None
    assert placeholder.preds == () and placeholder.succs == ()
    assert placeholder.children == ()
