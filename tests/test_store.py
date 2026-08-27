"""store.py regression tests: Op history JSON round-trip through the
full pipeline (reduce -> save -> load -> extract -> build)."""

from cfg_reducer import GraphEngine, ReductionAlgorithm, metagraph, motif, store

NESTED_LOOP = [
    ("A", "B"), ("B", "C"), ("C", "D"),
    ("D", "C"), ("D", "B"), ("B", "E"),
]


def _reduce(edges: list[tuple[str, str]]) -> GraphEngine:
    engine = GraphEngine()
    for n in sorted({n for e in edges for n in e}):
        engine.add_node(n)
    for src, dst in edges:
        engine.add_edge(src, dst)

    algorithm = ReductionAlgorithm(engine)
    while algorithm.step() is not None:
        pass
    return engine


def test_op_roundtrip_preserves_kinds_and_targets(tmp_path):
    engine = _reduce(NESTED_LOOP)
    path = tmp_path / "ops.json"

    store.save(engine.history, path)
    loaded = store.load(path)

    assert [op.kind for op in loaded] == [op.kind for op in engine.history]
    assert [op.forward.get("target") for op in loaded] == \
           [op.forward.get("target") for op in engine.history]


def test_roundtrip_yields_identical_metagraph(tmp_path):
    engine = _reduce(NESTED_LOOP)
    path = tmp_path / "ops.json"
    store.save(engine.history, path)

    direct = metagraph.build(motif.extract(engine.history))
    loaded = metagraph.build(motif.extract(store.load(path)))

    assert loaded == direct
