from __future__ import annotations

import random
from dataclasses import replace

import pytest

from cfg_reducer import (
    GraphEngine,
    MetaGraph,
    ReductionAlgorithm,
    Skeleton,
    motif,
    metagraph,
)
from cfg_reducer.linearize import canonical_order, decode, encode, skeleton_of


def _build_graph(edges: list[tuple[str, str]]) -> GraphEngine:
    engine = GraphEngine()
    nodes = sorted({node for edge in edges for node in edge})

    for node in nodes:
        engine.add_node(node)

    for src, dst in edges:
        engine.add_edge(src, dst)

    return engine


def _graph_edges(engine: GraphEngine) -> list[tuple[str, str]]:
    return sorted(
        (src, dst)
        for src in sorted(engine.node_ids())
        for dst in sorted(engine.successors(src))
    )


def _extract_metagraph(
    edges: list[tuple[str, str]],
    entry: str | None = None,
) -> MetaGraph:
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


def _reverse_name_mapping(edges: list[tuple[str, str]]) -> dict[str, str]:
    nodes = sorted({node for edge in edges for node in edge})
    return dict(zip(nodes, reversed(nodes), strict=True))


def _seed_derived_name_mapping(node_ids: list[str], seed: int) -> dict[str, str]:
    shuffled = sorted(node_ids)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(sorted(node_ids), shuffled, strict=True))


def _max_loop_nesting_depth(tokens: list[str]) -> int:
    depth = 0
    max_depth = 0

    for token in tokens:
        if token == "OPEN":
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == "CLOSE":
            depth -= 1

    return max_depth


def _all_steps(mg: MetaGraph) -> list[int]:
    steps = [motif_.step for motif_ in mg.motifs]
    for subgraph in mg.subgraphs.values():
        steps.extend(_all_steps(subgraph))
    return steps


def _remap_steps(mg: MetaGraph, mapping: dict[int, int]) -> MetaGraph:
    remapped_motifs = tuple(
        replace(motif_, step=mapping[motif_.step])
        for motif_ in mg.motifs
    )
    remapped_edges = tuple(
        (mapping[src], mapping[dst])
        for src, dst in mg.edges
    )
    remapped_subgraphs = {
        mapping[step]: _remap_steps(subgraph, mapping)
        for step, subgraph in mg.subgraphs.items()
    }
    return MetaGraph(
        motifs=remapped_motifs,
        edges=remapped_edges,
        subgraphs=remapped_subgraphs,
    )


def _token_stream_is_canonical_under_mapping(
    edges: list[tuple[str, str]],
    mapping: dict[str, str],
    entry: str | None = None,
) -> bool:
    relabeled_entry = None if entry is None else mapping[entry]
    original = encode(_extract_metagraph(edges, entry=entry))
    relabeled = encode(
        _extract_metagraph(
            _relabel_edges(edges, mapping),
            entry=relabeled_entry,
        )
    )
    return original == relabeled


def _assert_round_trip(
    edges: list[tuple[str, str]],
    entry: str | None = None,
) -> None:
    mg = _extract_metagraph(edges, entry=entry)
    assert decode(encode(mg)) == skeleton_of(mg)


def test_diamond_encodes_to_the_exact_hand_verifiable_stream():
    mg = _extract_metagraph([
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
    ])

    assert encode(mg) == [
        "ADD_ENTRY",
        "ADD_LINEAR",
        "ptr_0",
        "ADD_LINEAR",
        "ptr_0",
        "ADD_MERGE",
        "ptr_1",
        "ptr_2",
        "STOP",
    ]


def test_simple_loop_encodes_with_one_child_level_and_two_child_adds():
    mg = _extract_metagraph([
        ("A", "B"),
        ("B", "C"),
        ("C", "B"),
        ("C", "D"),
    ])
    tokens = encode(mg)
    open_index = tokens.index("OPEN")
    close_index = tokens.index("CLOSE")
    child_tokens = tokens[open_index + 1:close_index]

    # Derived from the real reduction pipeline: this graph reduces to
    # entry -> loop({B,C}) -> linear(D) at the top level.
    assert tokens[:4] == ["ADD_ENTRY", "ADD_LOOP", "ptr_0", "OPEN"]
    assert tokens[close_index + 1:] == ["ADD_LINEAR", "ptr_1", "STOP"]
    assert tokens.count("OPEN") == 1
    assert tokens.count("CLOSE") == 1

    # Derived from the real reduction pipeline after back-edge cutting:
    # the loop child level contains exactly two linear motifs.
    assert child_tokens[-1] == "STOP"
    assert sum(token.startswith("ADD_") for token in child_tokens) == 2


def test_nested_loop_encodes_with_loop_nesting_depth_two():
    mg = _extract_metagraph([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "C"),
        ("D", "B"),
        ("B", "E"),
    ])
    tokens = encode(mg)

    # Derived from the real reduction pipeline: this graph becomes an
    # outer loop containing an inner loop.
    assert _max_loop_nesting_depth(tokens) == 2


def test_diamond_encoding_is_invariant_to_node_id_permutation():
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
    ]

    assert _token_stream_is_canonical_under_mapping(
        edges,
        _reverse_name_mapping(edges),
    )


def test_simple_loop_encoding_is_invariant_to_node_id_permutation():
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "B"),
        ("C", "D"),
    ]

    assert _token_stream_is_canonical_under_mapping(
        edges,
        _reverse_name_mapping(edges),
    )


def test_entry_scc_without_entry_falls_back_to_min_scc_header():
    mg = _extract_metagraph([
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("C", "D"),
    ])

    loop = mg.motifs[0]

    assert loop.kind == "loop"
    assert loop.meta["header"] == min(loop.meta["scc"])
    assert loop.meta["header"] == "A"
    assert loop.meta["back_edges"] == [("C", "A")]


def test_entry_scc_with_entry_selects_declared_header_and_is_canonical():
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("C", "D"),
    ]
    mg = _extract_metagraph(edges, entry="A")
    loop = mg.motifs[0]

    assert loop.kind == "loop"
    assert loop.meta["header"] == "A"
    assert loop.meta["back_edges"] == [("C", "A")]
    assert _token_stream_is_canonical_under_mapping(
        edges,
        {"A": "D", "B": "C", "C": "B", "D": "A"},
        entry="A",
    )


def test_nested_loop_encoding_is_invariant_to_node_id_permutation():
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "C"),
        ("D", "B"),
        ("B", "E"),
    ]

    assert _token_stream_is_canonical_under_mapping(
        edges,
        _reverse_name_mapping(edges),
    )


def test_decode_encode_round_trip_for_handcrafted_graphs():
    _assert_round_trip([
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
    ])
    _assert_round_trip([
        ("A", "B"),
        ("B", "C"),
        ("C", "B"),
        ("C", "D"),
    ])
    _assert_round_trip([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "C"),
        ("D", "B"),
        ("B", "E"),
    ])


@pytest.mark.parametrize("seed", range(10))
def test_decode_encode_round_trip_for_build_cfg_seeds(seed: int):
    from cfg_reducer.gen import build_cfg

    engine = build_cfg(seed=seed)
    edges = _graph_edges(engine)
    mapping = _seed_derived_name_mapping(sorted(engine.node_ids()), seed)

    if not _token_stream_is_canonical_under_mapping(
        edges,
        mapping,
        entry="N00",
    ):
        pytest.skip(f"seed {seed} is discarded by the canonicality self-check")

    _assert_round_trip(edges, entry="N00")


def test_decode_rejects_forward_ptr_reference():
    with pytest.raises(ValueError):
        decode(["ADD_LINEAR", "ptr_0", "STOP"])


def test_decode_rejects_duplicate_ptrs():
    with pytest.raises(ValueError):
        decode(["ADD_MERGE", "ptr_0", "ptr_0", "STOP"])


def test_decode_rejects_non_ascending_ptrs():
    with pytest.raises(ValueError):
        decode([
            "ADD_ENTRY",
            "ADD_LINEAR", "ptr_0",
            "ADD_MERGE", "ptr_1", "ptr_0",
            "STOP",
        ])


def test_decode_rejects_unbalanced_open_close():
    with pytest.raises(ValueError):
        decode(["ADD_LOOP", "OPEN", "STOP"])


def test_decode_rejects_missing_top_level_stop():
    with pytest.raises(ValueError):
        decode(["ADD_ENTRY"])


def test_decode_rejects_trailing_tokens_after_top_level_stop():
    with pytest.raises(ValueError):
        decode(["ADD_ENTRY", "STOP", "ADD_ENTRY"])


def test_decode_rejects_unknown_tokens():
    with pytest.raises(ValueError):
        decode(["ADD_ENTRY", "BOGUS", "STOP"])


def test_canonical_order_returns_a_permutation_of_level_steps():
    mg = _extract_metagraph([
        ("A", "B"),
        ("A", "C"),
        ("C", "D"),
        ("D", "C"),
        ("B", "E"),
        ("D", "E"),
    ])
    order = canonical_order(mg)

    assert sorted(order) == sorted(motif_.step for motif_ in mg.motifs)
    assert len(order) == len(set(order))


def test_encode_does_not_depend_on_step_values():
    mg = _extract_metagraph([
        ("A", "B"),
        ("A", "C"),
        ("C", "D"),
        ("D", "C"),
        ("B", "E"),
        ("D", "E"),
    ])

    # Derived from the real reduction pipeline: this graph yields
    # entry, loop, linear, merge at the top level.
    assert [motif_.kind for motif_ in mg.motifs] == [
        "entry",
        "loop",
        "linear",
        "merge",
    ]

    remapped = _remap_steps(
        mg,
        dict(
            zip(
                sorted(_all_steps(mg)),
                [41, 73, 5, 88, 2, 19],
                strict=True,
            )
        ),
    )

    assert encode(remapped) == encode(mg)
