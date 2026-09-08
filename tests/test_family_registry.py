"""Plugin contracts and persisted v2 descriptor replay."""

import json
from pathlib import Path
from random import Random
import subprocess
import sys
from typing import cast

import pytest

from cfg_reducer import GraphEngine
from cfg_reducer import family_registry as registry
from cfg_reducer.dataset import build_dataset, cfg_edges
from cfg_reducer.generator_types import CFGShape, GeneratorSpec, Json
from cfg_reducer.generate_v2 import (
    descriptor_for, generate_cfg_v2, normalize_spec, spec_from_json, spec_to_json,
)
from cfg_reducer.store import sample_id_for


class Toy:
    def __init__(self, name: str = "toy") -> None:
        self.name = name

    def normalize(self, spec: GeneratorSpec) -> GeneratorSpec:
        if set(spec.params) - {"custom"}:
            raise ValueError("unknown toy params")
        return GeneratorSpec(spec.family, spec.num_nodes,
                             {"custom": spec.params.get("custom", {"z": [1], "a": True})})

    def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
        nodes = tuple(f"N{i:02d}" for i in range(spec.num_nodes))
        edges = list(zip(nodes, nodes[1:]))
        if len(nodes) > 2 and rng.random() < 0.5:
            edges.append((nodes[0], nodes[-1]))
        return CFGShape(nodes, tuple(reversed(edges)), nodes[0])


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_families", {})


def test_registry_contract() -> None:
    toy = Toy()
    registry.register_family(toy)
    registry.register_family(Toy("another"))
    assert registry.family_names() == ("another", "toy")
    assert registry.get_family("toy") is toy
    with pytest.raises(ValueError, match="unknown"):
        registry.get_family("missing")
    for duplicate in (toy, Toy()):
        with pytest.raises(ValueError, match="duplicate"):
            registry.register_family(duplicate)
    params: dict[str, Json] = {"custom": {"z": [1, {"nested": "value"}], "a": False}}
    spec = GeneratorSpec("toy", 3, params)
    params.clear()
    normalized = normalize_spec(spec)
    assert normalize_spec(normalized) == normalized
    encoded = spec_to_json(normalized)
    assert spec_from_json(json.loads(json.dumps(encoded, allow_nan=False))) == normalized
    assert list(cast(dict[str, Json], normalized.params["custom"])) == ["a", "z"]
    cast(dict[str, Json], encoded["params"]).clear()
    assert normalized.params
    cast(dict[str, Json], spec.params["custom"])["z"] = None
    assert normalized.params != spec.params


def test_plugin_descriptor_roundtrip(tmp_path: Path) -> None:
    sample_ids = []
    for name in ("toy", "other_toy"):
        registry.register_family(Toy(name))
        spec = GeneratorSpec(name, 4)
        descriptor = descriptor_for(spec)
        assert descriptor.name == "cfg_v2:" + name
        config = {"spec": spec_to_json(normalize_spec(spec))}
        manifest = build_dataset(tmp_path / name, {"test": (7, 8)}, config,
                                 "test-v2", generator=descriptor)
        sid = manifest["splits"]["test"]["samples"][0]["sample_id"]
        payload = json.loads((tmp_path / name / "test" / f"{sid}.json").read_text())
        assert sample_id_for(payload["provenance"]) == sid
        sample_ids.append(sid)
        replay = GraphEngine()
        descriptor.fn(replay, seed=7, **payload["provenance"]["generator"]["config"])
        direct = GraphEngine()
        assert generate_cfg_v2(direct, seed=7, spec=spec) == list(replay.nodes)
        assert cfg_edges(direct) == cfg_edges(replay)
    assert sample_ids[0] != sample_ids[1]
    with pytest.raises(ValueError, match="family must match"):
        descriptor_for(GeneratorSpec("toy", 4)).fn(
            GraphEngine(), seed=7, spec=spec_to_json(GeneratorSpec("other_toy", 4)))
    with pytest.raises(ValueError, match="unknown toy params"):
        spec_from_json({"family": "toy", "num_nodes": 4, "params": {"unknown": 1}})


@pytest.mark.parametrize("value", [
    {"family": "toy", "num_nodes": True, "params": {}},
    {"family": "toy", "num_nodes": 2.0, "params": {}},
    {"family": "Toy", "num_nodes": 2, "params": {}},
    {"family": "toy", "num_nodes": 0, "params": {}},
    {"family": "toy", "num_nodes": 2, "params": [],},
    {"family": "toy", "num_nodes": 2, "params": {}, "extra": 1},
    {"family": "toy", "num_nodes": 2},
    {"family": "toy", "num_nodes": 2, "params": {"custom": float("nan")}},
    {"family": "toy", "num_nodes": 2, "params": {"custom": [float("inf")]}},
    {"family": "toy", "num_nodes": 2, "params": {"custom": -float("inf")}},
    {"family": "toy", "num_nodes": 2, "params": {1: "bad"}},
])
def test_invalid_json(value: object) -> None:
    registry.register_family(Toy())
    with pytest.raises(ValueError):
        spec_from_json(cast(dict[str, Json], value))


@pytest.mark.parametrize("shape", [
    CFGShape(("N00", "N00"), (), "N00"),
    CFGShape(("N00",), (("N00", "missing"),), "N00"),
    CFGShape(("N00",), (), "missing"),
    CFGShape(("N00",), (("N00", "N00"), ("N00", "N00")), "N00"),
])
def test_invalid_shape_leaves_engine_empty(shape: CFGShape) -> None:
    class Broken(Toy):
        def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
            return shape

    registry.register_family(Broken())
    engine = GraphEngine()
    with pytest.raises(ValueError):
        generate_cfg_v2(engine, seed=0, spec=GeneratorSpec("toy", 2))
    assert not engine.nodes


def test_dispatch_order_rng_and_registration_guard() -> None:
    class Checked(Toy):
        def generate(self, spec: GeneratorSpec, rng: Random) -> CFGShape:
            assert rng.random() == Random(42).random()
            with pytest.raises(ValueError, match="during generation"):
                registry.register_family(Toy("late"))
            return CFGShape(("N00", "N01", "N02"),
                            (("N01", "N02"), ("N00", "N01")), "N00")

    class RecordingEngine(GraphEngine):
        def __init__(self) -> None:
            super().__init__()
            self.added_edges: list[tuple[str, str]] = []

        def add_edge(self, src: str, dst: str) -> None:
            self.added_edges.append((src, dst))
            super().add_edge(src, dst)

    registry.register_family(Checked())
    engine = RecordingEngine()
    generate_cfg_v2(engine, seed=42, spec=GeneratorSpec("toy", 3))
    assert list(engine.nodes) == ["N00", "N01", "N02"]
    assert engine.added_edges == [("N00", "N01"), ("N01", "N02")]
    with pytest.raises(ValueError, match="empty engine"):
        generate_cfg_v2(engine, seed=42, spec=GeneratorSpec("toy", 3))
    registry.register_family(Toy("after"))


def test_load_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    for module, name in (("gr_test_plugin_a", "external_a"), ("gr_test_plugin_b", "external_b")):
        (tmp_path / f"{module}.py").write_text(
            "from cfg_reducer.family_registry import register_family, family_names\n"
            "from cfg_reducer.generator_types import CFGShape\n"
            + ("assert 'external_a' in family_names()\n" if name == "external_b" else "")
            + f"class Plugin:\n    name = {name!r}\n"
            "    def normalize(self, spec): return spec\n"
            "    def generate(self, spec, rng): return CFGShape(('N00',), (), 'N00')\n"
            "register_family(Plugin())\n")
        monkeypatch.delitem(sys.modules, module, raising=False)
    registry.load_plugins(("gr_test_plugin_a", "gr_test_plugin_b"))
    assert registry.family_names() == ("external_a", "external_b")
    registry.load_plugins(("gr_test_plugin_a",))  # Normal import caching.


def test_package_import_does_not_load_dataset() -> None:
    subprocess.run([sys.executable, "-c",
                    "import cfg_reducer, sys; assert 'cfg_reducer.dataset' not in sys.modules"],
                   check=True)
