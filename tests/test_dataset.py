"""Dataset generation tests: generator determinism, structural dedup,
split separation, and manifest/file layout."""

import json

import pytest

from cfg_reducer import GraphEngine, dataset, store
from cfg_reducer.generate import GENERATOR_NAME, generate_cfg

NUM_NODES = 8
EDGE_PROB = 0.3
CONFIG = {"num_nodes": NUM_NODES, "edge_prob": EDGE_PROB}


def _edges_for(seed: int) -> list[tuple[str, str]]:
    engine = GraphEngine()
    generate_cfg(engine, num_nodes=NUM_NODES, edge_prob=EDGE_PROB, seed=seed)
    return dataset.cfg_edges(engine)


def _constant_generator(engine, num_nodes, edge_prob, seed):
    """Ignores the seed: every call yields the same chain A -> B -> C."""
    for n in ("A", "B", "C"):
        engine.add_node(n)
    engine.add_edge("A", "B")
    engine.add_edge("B", "C")
    return ["A", "B", "C"]


# ── generator ────────────────────────────────

def test_generator_is_deterministic_per_seed():
    assert _edges_for(1) == _edges_for(1)
    assert any(_edges_for(s) != _edges_for(1) for s in range(2, 6))


def test_generated_cfgs_survive_the_full_pipeline():
    for seed in range(3):
        engine = GraphEngine()
        generate_cfg(engine, num_nodes=NUM_NODES, edge_prob=EDGE_PROB,
                     seed=seed)
        mg = dataset.reduce_to_metagraph(engine)

        assert engine.is_empty()
        payload = store.encode_sample(
            mg, {"source": "synthetic", "generator": {"seed": seed}}
        )
        assert store.decode_sample(payload) == mg


# ── structural dedup ─────────────────────────

def test_fingerprint_and_isomorphism_detect_relabeling():
    edges = _edges_for(0)
    relabeled = [(f"X{u}", f"X{v}") for u, v in edges]

    assert dataset.fingerprint(edges) == dataset.fingerprint(relabeled)
    assert dataset.is_structural_duplicate(edges, relabeled)


def test_distinct_structures_are_not_duplicates():
    chain = [("A", "B"), ("B", "C")]
    branch = [("A", "B"), ("A", "C")]
    assert not dataset.is_structural_duplicate(chain, branch)


# ── build_dataset ────────────────────────────

def test_build_dataset_layout_and_manifest(tmp_path):
    out = tmp_path / "ds"
    manifest = dataset.build_dataset(
        out, {"train": (0, 6), "val": (6, 9)}, CONFIG, version="test",
    )

    assert manifest == json.loads((out / "manifest.json").read_text())

    for split_name, (start, stop) in {"train": (0, 6), "val": (6, 9)}.items():
        info = manifest["splits"][split_name]
        assert info["kept"] + info["dropped_duplicates"] == stop - start

        for entry in info["samples"]:
            path = out / split_name / f"{entry['sample_id']}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["sample_id"] == entry["sample_id"]
            assert payload["provenance"]["generator"]["seed"] == entry["seed"]
            assert payload["sample_id"] == store.sample_id_for(
                payload["provenance"]
            )
            store.load_sample(path)   # decodes without error

    # A3-4: no structural overlap may survive across splits
    train_edges = [
        _edges_for(e["seed"]) for e in manifest["splits"]["train"]["samples"]
    ]
    val_edges = [
        _edges_for(e["seed"]) for e in manifest["splits"]["val"]["samples"]
    ]
    for ve in val_edges:
        assert not any(
            dataset.is_structural_duplicate(ve, te) for te in train_edges
        )


def test_build_dataset_drops_structural_duplicates(tmp_path):
    manifest = dataset.build_dataset(
        tmp_path / "ds", {"train": (0, 3), "val": (3, 5)}, CONFIG,
        version="test", generator=_constant_generator,
    )

    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    assert train["kept"] == 1
    assert train["dropped_duplicates"] == 2
    assert val["kept"] == 0
    assert val["dropped_duplicates"] == 2

    survivor = train["samples"][0]["sample_id"]
    assert all(
        d["duplicate_of"] == survivor
        for d in train["dropped"] + val["dropped"]
    )


def test_overlapping_seed_ranges_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        dataset.build_dataset(
            tmp_path / "ds", {"train": (0, 5), "val": (4, 8)},
            CONFIG, version="test",
        )


def test_cli_builds_dataset(tmp_path, capsys):
    out = tmp_path / "ds"
    dataset.main([
        "--out", str(out),
        "--split", "train=0:4",
        "--split", "val=4:6",
        "--num-nodes", "8",
        "--edge-prob", "0.3",
        "--version", "test",
    ])

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["generator"] == {
        "name": GENERATOR_NAME, "version": "test", "config": CONFIG,
    }
    assert "train: kept" in capsys.readouterr().out


# ── external review 2026-09-04: identity fixes ──

def test_isolated_nodes_count_toward_identity():
    edges = [("A", "B")]
    with_isolated = ["A", "B", "C"]
    plain = ["A", "B"]

    # edges-only identity (legacy call) cannot see the isolated node
    assert dataset.is_structural_duplicate(edges, edges)
    # node-aware identity distinguishes the two graphs
    assert dataset.fingerprint(edges, with_isolated) != dataset.fingerprint(edges, plain)
    assert not dataset.is_structural_duplicate(edges, edges, with_isolated, plain)
    assert dataset.is_structural_duplicate(edges, edges, with_isolated, with_isolated)


def test_cfg_nodes_snapshot_includes_isolated_nodes():
    engine = GraphEngine()
    for n in ("A", "B", "Z"):
        engine.add_node(n)
    engine.add_edge("A", "B")
    assert dataset.cfg_nodes(engine) == ["A", "B", "Z"]
    assert dataset.cfg_edges(engine) == [("A", "B")]


def test_generator_descriptor_name_is_recorded(tmp_path):
    desc = dataset.GeneratorDescriptor("const_chain", _constant_generator)
    manifest = dataset.build_dataset(
        tmp_path / "ds", {"train": (0, 1)}, CONFIG, version="test", generator=desc,
    )
    assert manifest["generator"]["name"] == "const_chain"
    sid = manifest["splits"]["train"]["samples"][0]["sample_id"]
    payload = json.loads((tmp_path / "ds" / "train" / f"{sid}.json").read_text())
    assert payload["provenance"]["generator"]["name"] == "const_chain"

    # a bare callable is named after the function, never after the default
    manifest2 = dataset.build_dataset(
        tmp_path / "ds2", {"train": (0, 1)}, CONFIG, version="test",
        generator=_constant_generator,
    )
    assert manifest2["generator"]["name"] == "_constant_generator"
    assert manifest2["generator"]["name"] != GENERATOR_NAME
    # different generator identity -> different sample_id for the same seed
    assert manifest2["splits"]["train"]["samples"][0]["sample_id"] != sid


def test_manifest_records_code_state(tmp_path):
    manifest = dataset.build_dataset(
        tmp_path / "ds", {"train": (0, 2)}, CONFIG, version="test",
        code={"commit": "abc1234", "dirty": True},
    )
    assert manifest["code"] == {"commit": "abc1234", "dirty": True}

    plain = dataset.build_dataset(
        tmp_path / "ds2", {"train": (0, 2)}, CONFIG, version="test",
    )
    assert "code" not in plain

    state = dataset._git_state()
    assert set(state) == {"commit", "dirty"}
    assert isinstance(state["commit"], str) and isinstance(state["dirty"], bool)
