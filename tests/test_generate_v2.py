"""Layered v1 compatibility and dataset integration."""

import json
import os
from pathlib import Path
from random import Random

import pytest

from cfg_reducer import GraphEngine
from cfg_reducer.dataset import build_dataset, cfg_edges, cfg_nodes, reduce_to_metagraph
from cfg_reducer.families.layered import Layered, _choose_target
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
