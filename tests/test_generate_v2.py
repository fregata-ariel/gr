"""Layered v1 compatibility and dataset integration."""

from collections import Counter
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys

import pytest

from cfg_reducer import GraphEngine
from cfg_reducer.dataset import build_dataset, cfg_edges, cfg_nodes, reduce_to_metagraph
from cfg_reducer.families.layered import Layered, _choose_target
from cfg_reducer.families.structured import (
    GenerationRejected, Structured, lower_structure, pad_shape, plan_structure,
)
from cfg_reducer.reducibility import is_reducible
from cfg_reducer.generate import generate_cfg
from cfg_reducer.generate_v2 import (
    descriptor_for, generate_cfg_v2, normalize_spec, spec_to_json,
)
from cfg_reducer.generator_types import GeneratorSpec, Json
from cfg_reducer.model_input import sketch_of


_SLOW = os.environ.get("GR_SLOW_TESTS") == "1"


@pytest.mark.parametrize("n", [1, 2, 4, 6, 12, 24, 32, 48])
@pytest.mark.parametrize("seed", range(50 if _SLOW else 10))
@pytest.mark.parametrize("p", [0, 0.18, 0.5, 1] if _SLOW else [0.18, 0.5])
def test_layered_v1_exact(n: int, seed: int, p: float) -> None:
    spec = GeneratorSpec("layered", n, {
        "max_layer_width": 3, "edge_prob": p,
        "loop_count": max(1, int(.15 * n)),
        "goto_count": max(1, int(.1 * n)), "span_mode": "uniform",
    })
    old, new = GraphEngine(), GraphEngine()
    assert generate_cfg(old, n, p, seed) == generate_cfg_v2(new, seed=seed, spec=spec)
    assert cfg_nodes(old) == cfg_nodes(new)
    assert cfg_edges(old) == cfg_edges(new)
    assert sketch_of(reduce_to_metagraph(old)) == sketch_of(reduce_to_metagraph(new))


@pytest.mark.parametrize("loops,gotos", [(1, 1), (1, 0), (0, 1)])
def test_layered_n3_error(loops: int, gotos: int) -> None:
    with pytest.raises(ValueError) as old:
        generate_cfg(GraphEngine(), 3, seed=7)
    with pytest.raises(ValueError) as new:
        generate_cfg_v2(GraphEngine(), seed=7, spec=GeneratorSpec("layered", 3, {
            "loop_count": loops, "goto_count": gotos,
        }))
    assert type(old.value) is type(new.value)


def test_layered_n3_dag() -> None:
    shape = Layered().generate(GeneratorSpec("layered", 3, {
        "loop_count": 0, "goto_count": 0,
    }), Random(7))
    assert shape.edges == (("N00", "N01"), ("N01", "N02"))


@pytest.mark.parametrize("n", [1, 2])
def test_layered_small_chain_no_random(n: int) -> None:
    rng = Random(7)
    before = rng.getstate()
    shape = Layered().generate(GeneratorSpec("layered", n), rng)
    assert shape.nodes == tuple(f"N{i:02d}" for i in range(n))
    assert shape.edges == tuple(zip(shape.nodes, shape.nodes[1:]))
    assert shape.entry == "N00"
    assert rng.getstate() == before


@pytest.mark.parametrize("mode", ["uniform", "short", "long"])
@pytest.mark.parametrize("length", range(1, 11))
def test_layered_span_targets(mode: str, length: int) -> None:
    lo, hi = 5, 5 + length - 1
    third = (length + 2) // 3
    candidates = list(range(lo, hi + 1))
    if mode == "short":
        candidates = candidates[:third]
    elif mode == "long":
        candidates = candidates[-third:]
    actual, expected = Random(42), Random(42)
    for _ in range(20):
        target = _choose_target(actual, lo, hi, mode)
        assert target in candidates
        assert target == (expected.randint(lo, hi) if mode == "uniform"
                          else expected.choice(candidates))
    assert actual.getstate() == expected.getstate()
    spec = GeneratorSpec("layered", 24, {"span_mode": mode})
    assert Layered().generate(spec, Random(42)) == Layered().generate(spec, Random(42))


@pytest.mark.parametrize("key", [
    "branch_degree", "merge_degree", "target_depth", "spaghetti_rate", "unknown",
])
def test_layered_unknown_params(key: str) -> None:
    with pytest.raises(ValueError, match="unknown layered params"):
        normalize_spec(GeneratorSpec("layered", 12, {key: 1}))


@pytest.mark.parametrize("key,value", [
    ("max_layer_width", 0), ("max_layer_width", True), ("max_layer_width", 1.5),
    ("loop_count", -1), ("loop_count", False), ("loop_count", 1.5),
    ("goto_count", -1), ("goto_count", True), ("goto_count", "1"),
    ("edge_prob", -0.1), ("edge_prob", 1.1), ("edge_prob", True),
    ("edge_prob", "0.5"), ("edge_prob", None), ("edge_prob", float("nan")),
    ("edge_prob", float("inf")), ("span_mode", "invalid"), ("span_mode", []),
])
def test_layered_invalid_params(key: str, value: Json) -> None:
    with pytest.raises(ValueError):
        normalize_spec(GeneratorSpec("layered", 12, {key: value}))


def test_layered_normalize() -> None:
    spec = GeneratorSpec("layered", 12)
    normalized = normalize_spec(spec)
    assert normalized.params == {
        "max_layer_width": 3, "edge_prob": 0.5, "loop_count": 2,
        "goto_count": 1, "span_mode": "uniform",
    }
    assert spec.params == {}
    assert normalize_spec(normalized) == normalized
    for p in (0, 1):
        boundary = GeneratorSpec("layered", 4, {
            "max_layer_width": 1, "edge_prob": p, "loop_count": 0, "goto_count": 0,
        })
        assert normalize_spec(normalize_spec(boundary)) == normalize_spec(boundary)
        assert Layered().generate(boundary, Random(0)).edges == (
            ("N00", "N01"), ("N01", "N02"), ("N02", "N03"),
        )
    with pytest.raises(ValueError):
        GeneratorSpec("layered", 0)


def test_layered_dataset_descriptor(tmp_path: Path) -> None:
    spec = GeneratorSpec("layered", 6)
    config = {"spec": spec_to_json(normalize_spec(spec))}
    descriptor = descriptor_for(spec)
    manifest = build_dataset(tmp_path, {"test": (7, 8)}, config,
                             "test-layered", generator=descriptor)
    samples = manifest["splits"]["test"]["samples"]
    assert len(samples) == 1
    payload = json.loads((tmp_path / "test" / f"{samples[0]['sample_id']}.json").read_text())
    generator = payload["provenance"]["generator"]
    assert generator["name"] == "cfg_v2:layered"
    assert generator["config"] == config
    replay, direct = GraphEngine(), GraphEngine()
    assert descriptor.fn(replay, seed=7, **generator["config"]) == generate_cfg_v2(
        direct, seed=7, spec=spec)
    assert cfg_edges(replay) == cfg_edges(direct)


@pytest.mark.parametrize('n', [12, 24, 48])
def test_structured_reducible_many_seeds(n: int) -> None:
    successes = rejected = 0
    for seed in range(500 if _SLOW else 100):
        spec = GeneratorSpec('structured', n, {
            'loop_count': 2, 'goto_count': 1, 'merge_degree': 2,
            'branch_degree': 4, 'max_layer_width': 4,
            'span_mode': ('short', 'long', 'uniform')[seed % 3]})
        try:
            shape = Structured().generate(spec, Random(seed))
        except GenerationRejected as exc:
            assert exc.reason == 'node_budget'
            with pytest.raises(GenerationRejected):
                Structured().generate(spec, Random(seed))
            rejected += 1
            continue
        successes += 1
        assert is_reducible(shape.nodes, shape.edges, entry=shape.nodes[0])
        assert len(shape.edges) == len(set(shape.edges))
        assert shape == Structured().generate(spec, Random(seed))
        assert (9*n+9)//10 <= len(shape.nodes) <= 11*n//10
        assert max(Counter(v for _, v in shape.edges).values()) <= 2
    assert successes > 0
    print(f'n={n}: successes={successes}, GenerationRejected={rejected}')


@pytest.mark.parametrize('mode', ['short', 'long', 'uniform'])
@pytest.mark.parametrize('merge', [2, 3, 4])
def test_node_budget(mode: str, merge: int) -> None:
    for seed in range(5):
        spec = GeneratorSpec('structured', 24, {'merge_degree': merge, 'span_mode': mode})
        before = lower_structure(plan_structure(spec, Random(seed)), merge)
        after = pad_shape(before, 26, mode, Random(seed))
        for shape in (before, after, Structured().generate(spec, Random(seed))):
            assert is_reducible(shape.nodes, shape.edges, entry=shape.entry)
            assert max(Counter(v for _, v in shape.edges).values()) <= merge
            assert len(set(shape.edges)) == len(shape.edges)
        assert len(after.nodes) == 26
        assert len(after.edges) - len(before.edges) == 26 - len(before.nodes)


def test_structured_rejection_and_serial() -> None:
    # The lower bound passes, but sequential loops and join routers may exceed it.
    spec = GeneratorSpec('structured', 10, {'merge_degree': 2})
    rejected = successes = 0
    for seed in range(30):
        try:
            Structured().generate(spec, Random(seed))
            successes += 1
        except GenerationRejected as exc:
            assert str(exc) == exc.reason == 'node_budget'
            rejected += 1
    assert rejected and successes
    for n in (3, 12):
        shape = Structured().generate(GeneratorSpec('structured', n, {
            'max_layer_width': 1, 'loop_count': 0, 'goto_count': 0}), Random(0))
        assert shape.edges == tuple(zip(shape.nodes, shape.nodes[1:]))


def test_structured_hashseed() -> None:
    script = '''
import json
from dataclasses import asdict
from random import Random
from cfg_reducer.families.structured import Structured
from cfg_reducer.generator_types import GeneratorSpec
print(json.dumps([asdict(Structured().generate(GeneratorSpec('structured', 24,
    {'span_mode': mode, 'branch_degree': 4, 'max_layer_width': 4}), Random(seed)))
    for mode in ('short', 'long', 'uniform') for seed in range(3)], sort_keys=True))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', script],
               env=os.environ | {'PYTHONHASHSEED': value}) for value in ('1', '77')]
    assert outputs[0] == outputs[1]


def test_structured_dataset_descriptor(tmp_path: Path) -> None:
    spec = GeneratorSpec('structured', 12)
    config = {'spec': spec_to_json(normalize_spec(spec))}
    descriptor = descriptor_for(spec)
    manifest = build_dataset(tmp_path, {'test': (7, 8)}, config,
                             'test-structured', generator=descriptor)
    samples = manifest['splits']['test']['samples']
    assert len(samples) == 1
    payload = json.loads((tmp_path / 'test' / f"{samples[0]['sample_id']}.json").read_text())
    generator = payload['provenance']['generator']
    assert generator['name'] == 'cfg_v2:structured'
    assert generator['config'] == config
    replay, direct = GraphEngine(), GraphEngine()
    assert descriptor.fn(replay, seed=7, **generator['config']) == generate_cfg_v2(
        direct, seed=7, spec=spec)
    assert cfg_edges(replay) == cfg_edges(direct)


@pytest.mark.parametrize('mode,expected', [
    ('short', [(0, 1), (1, 4), (2, 4), (3, 4)]),
    ('long', [(1, 2), (1, 3)]),
    ('uniform', [(0, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]),
])
def test_structured_padding_candidates(mode, expected):
    from cfg_reducer.generator_types import CFGShape

    class RecordingRandom(Random):
        def choice(self, seq):
            assert seq == expected
            return seq[0]

    nodes = tuple(f'N{i:02d}' for i in range(5))
    edges = ((0, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4))
    shape = CFGShape(nodes, tuple((nodes[u], nodes[v]) for u, v in edges), nodes[0])
    padded = pad_shape(shape, 6, mode, RecordingRandom())
    src, dst = expected[0]
    mapped = {i: f'N{i + (i >= dst):02d}' for i in range(5)}
    inserted = f'N{dst:02d}'
    assert padded.edges == tuple(sorted(
        [(mapped[u], mapped[v]) for u, v in edges if (u, v) != (src, dst)]
        + [(mapped[src], inserted), (inserted, mapped[dst])]))
    assert is_reducible(padded.nodes, padded.edges, entry=padded.entry)


@pytest.mark.parametrize('seed', range(10))
def test_spaghetti_rate_zero(seed: int) -> None:
    from cfg_reducer.families.spaghetti import Spaghetti

    params: dict[str, Json] = {'span_mode': ('short', 'long', 'uniform')[seed % 3]}
    base = Structured().generate(GeneratorSpec('structured', 24, params), Random(seed))
    spec = GeneratorSpec('spaghetti', 24, params | {'spaghetti_rate': 0})
    assert Spaghetti().generate(spec, Random(seed)) == base
    assert descriptor_for(spec).name == 'cfg_v2:spaghetti'
    engine = GraphEngine()
    assert generate_cfg_v2(engine, seed=seed, spec=spec) == list(base.nodes)
    assert tuple(cfg_edges(engine)) == base.edges


@pytest.mark.parametrize('n', [3, 4, 24])
@pytest.mark.parametrize('rate', [0.1, 0.5, 1.0])
def test_spaghetti_edge_budget(n: int, rate: float) -> None:
    import math
    from cfg_reducer.families.spaghetti import Spaghetti

    params: dict[str, Json] = {'loop_count': 0, 'goto_count': 0} if n < 12 else {}
    for seed in range(5):
        rng = Random(seed)
        base = Structured().generate(GeneratorSpec('structured', n, params), rng)
        candidates = sorted((u, v) for u in base.nodes for v in base.nodes
                            if u != v and (u, v) not in base.edges
                            and v != base.entry and u != base.nodes[-1])
        budget = min(math.floor(rate * len(base.edges)), len(candidates))
        expected = rng.sample(candidates, budget)
        spec = GeneratorSpec('spaghetti', n, params | {'spaghetti_rate': rate})
        shape = Spaghetti().generate(spec, Random(seed))
        extra = set(shape.edges) - set(base.edges)
        assert len(extra) == budget
        assert extra == set(expected)
        assert set(base.edges) <= set(shape.edges)
        assert shape.edges == tuple(sorted(set(shape.edges)))
        assert shape.nodes == base.nodes and shape.entry == base.entry
        assert all(u != v and v != shape.entry and u != shape.nodes[-1] for u, v in extra)
        reached = {shape.entry}
        while True:
            expanded = reached | {v for u, v in shape.edges if u in reached}
            if expanded == reached:
                break
            reached = expanded
        assert reached == set(shape.nodes)


@pytest.mark.parametrize('rate', [-0.1, 1.1, True, None, '0.1', float('nan'), float('inf')])
def test_spaghetti_invalid_rate(rate: Json) -> None:
    with pytest.raises(ValueError):
        normalize_spec(GeneratorSpec('spaghetti', 24, {'spaghetti_rate': rate}))


def test_spaghetti_normalize() -> None:
    from cfg_reducer.generator_types import GenerationRejected as SharedRejected

    assert GenerationRejected is SharedRejected
    normalized = normalize_spec(GeneratorSpec('spaghetti', 24))
    assert normalized.params == normalize_spec(GeneratorSpec('structured', 24)).params | {
        'spaghetti_rate': 0.1}
    assert normalize_spec(normalized) == normalized
    invalid: list[tuple[str, dict[str, Json]]] = [
        ('structured', {'spaghetti_rate': 0}),
        ('spaghetti', {'unknown': 1}),
        ('spaghetti', {'merge_degree': 1}),
    ]
    for family, params in invalid:
        with pytest.raises(ValueError):
            normalize_spec(GeneratorSpec(family, 24, params))
