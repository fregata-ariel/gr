"""model_input tests: flatten rows, tokenize/detokenize round-trips,
and strict grammar validation."""

import pytest

from cfg_reducer import (
    GraphEngine, MetaGraph, Motif, ReductionAlgorithm,
    metagraph, model_input, motif,
)
from cfg_reducer.generate import generate_cfg

DIAMOND = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
SIMPLE_LOOP = [("A", "B"), ("B", "C"), ("C", "B"), ("C", "D")]
NESTED_LOOP = [
    ("A", "B"), ("B", "C"), ("C", "D"),
    ("D", "C"), ("D", "B"), ("B", "E"),
]


def _reduce_engine(engine: GraphEngine) -> MetaGraph:
    algorithm = ReductionAlgorithm(engine)
    while algorithm.step() is not None:
        pass
    return metagraph.build(motif.extract(engine.history))


def _metagraph(edges: list[tuple[str, str]]) -> MetaGraph:
    engine = GraphEngine()
    for n in sorted({n for e in edges for n in e}):
        engine.add_node(n)
    for src, dst in edges:
        engine.add_edge(src, dst)
    return _reduce_engine(engine)


def _mg_from_seed(seed: int) -> MetaGraph:
    engine = GraphEngine()
    generate_cfg(engine, num_nodes=12, edge_prob=0.18, seed=seed)
    return _reduce_engine(engine)


def _ids(names: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab[n] for n in names]


# ── flatten ──────────────────────────────────

def test_flatten_nested_loop_rows():
    rows = model_input.flatten(_metagraph(NESTED_LOOP))

    assert [r["step"] for r in rows] == [0, 5, 1, 4, 2, 3, 6]
    assert [r["kind"] for r in rows] == [
        "entry", "loop", "linear", "loop", "linear", "linear", "linear",
    ]
    assert [r["parent_loop_step"] for r in rows] == [
        None, None, 5, 5, 4, 4, None,
    ]
    assert [r["depth"] for r in rows] == [0, 0, 1, 1, 2, 2, 0]
    assert [r["position"] for r in rows] == [0, 1, 0, 1, 0, 1, 2]
    assert [r["in_offsets"] for r in rows] == [
        [], [1], [], [1], [], [1], [1],
    ]


def test_max_offset_needed():
    assert model_input.max_offset_needed(_metagraph(NESTED_LOOP)) == 1
    assert model_input.max_offset_needed(_metagraph(DIAMOND)) == 2


# ── vocabulary ───────────────────────────────

def test_vocab_is_deterministic_with_pad_zero():
    vocab = model_input.build_vocab(3)
    assert vocab == model_input.build_vocab(3)
    assert vocab["PAD"] == 0
    assert {f"REF_{k}" for k in (1, 2, 3)} <= vocab.keys()

    with pytest.raises(ValueError):
        model_input.build_vocab(0)


# ── tokenize / detokenize ────────────────────

def test_nested_loop_token_stream_matches_design_doc():
    mg = _metagraph(NESTED_LOOP)
    vocab = model_input.build_vocab(1)
    names = {i: t for t, i in vocab.items()}

    stream = [names[i] for i in model_input.tokenize(mg, vocab)]
    assert stream == [
        "BOS",
        "KIND_ENTRY",
        "KIND_LOOP", "REF_1", "LOOP_START",
        "KIND_LINEAR",
        "KIND_LOOP", "REF_1", "LOOP_START",
        "KIND_LINEAR",
        "KIND_LINEAR", "REF_1",
        "LOOP_END",
        "LOOP_END",
        "KIND_LINEAR", "REF_1",
        "EOS",
    ]


@pytest.mark.parametrize("edges", [DIAMOND, SIMPLE_LOOP, NESTED_LOOP])
def test_round_trip_on_fixed_graphs(edges):
    mg = _metagraph(edges)
    vocab = model_input.build_vocab(max(1, model_input.max_offset_needed(mg)))

    tokens = model_input.tokenize(mg, vocab)
    assert model_input.detokenize(tokens, vocab) == model_input.sketch_of(mg)


def test_round_trip_on_generated_cfgs():
    for seed in range(6):
        mg = _mg_from_seed(seed)
        vocab = model_input.build_vocab(
            max(1, model_input.max_offset_needed(mg))
        )
        tokens = model_input.tokenize(mg, vocab)

        assert all(0 <= t < len(vocab) for t in tokens)
        assert model_input.detokenize(tokens, vocab) == \
            model_input.sketch_of(mg)


def test_unknown_kind_rejected():
    stray = MetaGraph(
        motifs=(Motif(kind="annotate", node=None, preds=(), succs=()),),
        edges=(),
    )
    with pytest.raises(ValueError, match="unknown motif kind"):
        model_input.tokenize(stray, model_input.build_vocab(1))


def test_offset_beyond_vocab_window_rejected():
    mg = _metagraph(DIAMOND)     # needs REF_2
    with pytest.raises(ValueError, match="exceeds the vocab window"):
        model_input.tokenize(mg, model_input.build_vocab(1))


# ── grammar validation (model-output checking) ──

@pytest.mark.parametrize("names,message", [
    (["KIND_ENTRY", "EOS"], "BOS"),
    (["BOS", "KIND_ENTRY"], "BOS"),
    (["BOS", "REF_1", "EOS"], "no preceding motif"),
    (["BOS", "KIND_ENTRY", "LOOP_START", "LOOP_END", "EOS"],
     "must follow a loop motif"),
    (["BOS", "KIND_LOOP", "EOS"], "missing its LOOP_START"),
    (["BOS", "KIND_LOOP", "LOOP_START", "EOS"], "unbalanced LOOP_START"),
    (["BOS", "KIND_ENTRY", "LOOP_END", "EOS"], "unbalanced LOOP_END"),
    (["BOS", "KIND_ENTRY", "KIND_LINEAR", "REF_2", "EOS"],
     "before the start of its level"),
    (["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_MERGE", "REF_1", "REF_1",
      "EOS"], "strictly increasing"),
    (["BOS", "KIND_LOOP", "LOOP_START", "LOOP_END", "REF_1", "EOS"],
     "no preceding motif"),
    (["BOS", "PAD", "EOS"], "unexpected PAD"),
])
def test_malformed_streams_rejected(names, message):
    vocab = model_input.build_vocab(2)
    with pytest.raises(ValueError, match=message):
        model_input.detokenize(_ids(names, vocab), vocab)


def test_unknown_token_id_rejected():
    vocab = model_input.build_vocab(1)
    with pytest.raises(ValueError, match="unknown token id"):
        model_input.detokenize([vocab["BOS"], 999, vocab["EOS"]], vocab)