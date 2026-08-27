"""GraphEngine regression tests: Op execution, undo/redo, state restoration."""

from cfg_reducer import GraphEngine, Op


def _chain_engine() -> GraphEngine:
    """A -> B -> C"""
    engine = GraphEngine()
    for n in ("A", "B", "C"):
        engine.add_node(n)
    engine.add_edge("A", "B")
    engine.add_edge("B", "C")
    return engine


def _remove_node_op(engine: GraphEngine, target: str) -> Op:
    """Build a remove_node Op the same way ReductionAlgorithm does."""
    node = engine.nodes[target]
    preds = sorted(node.pred)
    return Op(
        kind="remove_node",
        forward={"target": target},
        inverse={
            "target": target,
            "pred_edges": preds,
            "succ_edges": sorted(node.succ),
            "weight": node.weight,
            "node_type": node.node_type,
            "weight_deltas": {
                p: node.weight for p in preds if engine.has_node(p)
            },
        },
    )


def _snapshot(engine: GraphEngine) -> dict:
    return {
        nid: (set(n.pred), set(n.succ), n.weight, n.node_type)
        for nid, n in engine.nodes.items()
    }


def test_remove_node_detaches_and_propagates_weight():
    engine = _chain_engine()
    engine.execute(_remove_node_op(engine, "C"))

    assert not engine.has_node("C")
    assert engine.nodes["B"].succ == set()
    assert engine.nodes["B"].weight == 2   # own 1 + propagated 1 from C


def test_undo_restores_graph_exactly():
    engine = _chain_engine()
    before = _snapshot(engine)

    op = _remove_node_op(engine, "C")
    engine.execute(op)
    assert engine.undo() is op
    assert _snapshot(engine) == before
    assert engine.undo() is None


def test_redo_reapplies_op():
    engine = _chain_engine()
    engine.execute(_remove_node_op(engine, "C"))
    after_execute = _snapshot(engine)

    engine.undo()
    engine.redo()
    assert _snapshot(engine) == after_execute
    assert engine.redo() is None


def test_execute_discards_redo_future():
    engine = _chain_engine()
    engine.execute(_remove_node_op(engine, "C"))
    engine.undo()

    replacement = _remove_node_op(engine, "B")
    engine.execute(replacement)

    assert engine.history == [replacement]
    assert engine.cursor == 1
    assert engine.redo() is None


def test_remove_edges_roundtrip():
    engine = _chain_engine()
    before = _snapshot(engine)

    op = Op(
        kind="remove_edges",
        forward={"edges": [("A", "B")]},
        inverse={"edges": [("A", "B")]},
    )
    engine.execute(op)
    assert engine.nodes["A"].succ == set()
    assert engine.nodes["B"].pred == set()

    engine.undo()
    assert _snapshot(engine) == before
