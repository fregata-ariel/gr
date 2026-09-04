"""store.py regression tests: Op history JSON round-trip through the
full pipeline (reduce -> save -> load -> extract -> build) and the
canonical MetaGraph sample codec (docs/design/metagraph_schema.md)."""

import json
import uuid
from pathlib import Path

import pytest

from cfg_reducer import (
    GraphEngine, MetaGraph, Motif, ReductionAlgorithm, metagraph, motif, store,
)

FIXTURE = Path(__file__).parent / "fixtures" / "metagraph_nested_loop.json"

GENERATOR = {
    "name": "test-gen", "version": "0.0.0", "seed": 7,
    "config": {"nodes": 5},
}
PROVENANCE = {"source": "synthetic", "generator": GENERATOR}

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


# ── MetaGraph sample codec ───────────────────

def _metagraph(edges: list[tuple[str, str]]) -> MetaGraph:
    engine = _reduce(edges)
    return metagraph.build(motif.extract(engine.history))


def test_sample_roundtrip_preserves_metagraph():
    mg = _metagraph(NESTED_LOOP)
    payload = store.encode_sample(mg, PROVENANCE)
    assert store.decode_sample(payload) == mg


def test_sample_file_roundtrip(tmp_path):
    mg = _metagraph(NESTED_LOOP)
    path = tmp_path / "sample.json"
    store.save_sample(mg, PROVENANCE, path)
    assert store.load_sample(path) == mg


def test_encode_is_deterministic_across_provenance_key_order():
    mg = _metagraph(NESTED_LOOP)
    reordered = {k: PROVENANCE[k] for k in reversed(list(PROVENANCE))}

    a = json.dumps(store.encode_sample(mg, PROVENANCE))
    b = json.dumps(store.encode_sample(mg, reordered))
    assert a == b


def test_fixture_decodes_to_live_pipeline_output():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert store.decode_sample(payload) == _metagraph(NESTED_LOOP)


def test_sample_id_is_stable_uuid5():
    sid = store.sample_id_for(PROVENANCE)
    assert uuid.UUID(sid).version == 5
    assert store.sample_id_for(PROVENANCE) == sid

    changed = {**PROVENANCE, "generator": {**GENERATOR, "seed": 8}}
    assert store.sample_id_for(changed) != sid


def test_unknown_kind_rejected_on_encode():
    stray = Motif(kind="annotate", node=None, preds=(), succs=())
    with pytest.raises(ValueError, match="unknown motif kind"):
        store.encode_metagraph(MetaGraph(motifs=(stray,), edges=()))


def test_unknown_kind_rejected_on_decode():
    payload = store.encode_sample(_metagraph(NESTED_LOOP), PROVENANCE)
    payload["metagraph"]["motifs"][0]["kind"] = "annotate"
    with pytest.raises(ValueError, match="unknown motif kind"):
        store.decode_sample(payload)


def test_wrong_schema_version_rejected():
    payload = store.encode_sample(_metagraph(NESTED_LOOP), PROVENANCE)
    payload["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        store.decode_sample(payload)


def test_orphan_subgraph_rejected():
    payload = store.encode_sample(_metagraph(NESTED_LOOP), PROVENANCE)
    payload["metagraph"]["subgraphs"][0]["loop_step"] = 99
    with pytest.raises(ValueError, match="no matching"):
        store.decode_sample(payload)
