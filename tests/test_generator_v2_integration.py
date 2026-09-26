"""Small, torch-free integration of generator plugins and family OOD evaluation."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cfg_reducer import family_registry, store
from cfg_reducer.dataset import build_dataset
from cfg_reducer.dataset_v2 import main
from training import controlled_eval
from training.data_utils import read_jsonl, write_jsonl
from training.prepare_tokens import prepare


def test_v2_ood_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep built-ins, but undo the external registration and import after this test.
    monkeypatch.setattr(family_registry, "_families", dict(family_registry._families))
    module = "gr_integration_toy_plugin"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    (tmp_path / f"{module}.py").write_text(
        "from cfg_reducer.family_registry import register_family\n"
        "from cfg_reducer.generator_types import CFGShape\n"
        "class Toy:\n"
        "    name = 'integration_toy'\n"
        "    def normalize(self, spec): return spec\n"
        "    def generate(self, spec, rng):\n"
        "        nodes = tuple(f'N{i:02d}' for i in range(spec.num_nodes))\n"
        "        return CFGShape(nodes, tuple(zip(nodes, nodes[1:])), nodes[0])\n"
        "register_family(Toy())\n",
        encoding="utf-8",
    )
    family_registry.load_plugins((module,))
    # Record the imported module with monkeypatch so teardown removes it as well.
    plugin = sys.modules.pop(module)
    monkeypatch.setitem(sys.modules, module, plugin)
    assert "integration_toy" in family_registry.family_names()

    def generate(name: str, family: str, splits: list[str], *extra: str) -> dict:
        spec = tmp_path / f"{name}_spec.json"
        spec.write_text(json.dumps({"family": family, "num_nodes": 10, "params": {}}))
        args = ["--spec", str(spec), "--out", str(tmp_path / name),
                "--version", "integration"]
        for split in splits:
            args.extend(["--split", split])
        main([*args, *extra])
        return json.loads((tmp_path / name / "manifest.json").read_bytes())

    toy = generate("toy", "integration_toy", ["test=0:1"])
    assert toy["generator"]["name"] == "cfg_v2:integration_toy"
    assert toy["splits"]["test"]["accepted"] == 1

    source, target, bundle = (tmp_path / name for name in ("source", "target", "tokens"))
    source_manifest = generate("source", "layered", ["train=0:30", "val=30:36"])
    assert all(info["accepted"] > 0 for info in source_manifest["splits"].values())
    source_id = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    # Four accepted graphs fill two small buckets; both must reach evaluation.
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"test": {
        "dims": [{"feature": "mean_offset", "cuts": [1.1], "labels": ["near", "far"]}],
        "target_per_bucket": 2,
    }}))
    target_manifest = generate("target", "structured", ["test=0:12"],
                               "--exclude-dataset", str(source), "--plan", str(plan))
    assert target_manifest["selection"]["excluded_datasets"] == [source_id]
    assert target_manifest["splits"]["test"]["complete"]
    accepted = target_manifest["splits"]["test"]["samples"]
    assert len(accepted) == 4
    assert {entry["bucket"] for entry in accepted} == {"mean_offset=near", "mean_offset=far"}

    meta = prepare(source, bundle, window_from="train", test_dataset=target)
    index = json.loads((bundle / "evaluation_index.json").read_bytes())
    assert index["sources"] == {
        "train": source_id, "val": source_id,
        "test": hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest(),
    }
    assert index["selection"] == target_manifest["selection"]
    assert {sid: entry for sid, entry in index["samples"].items() if entry["split"] == "test"} == {
        entry["sample_id"]: {"split": "test", "bucket": entry["bucket"],
                             "realized": entry["realized"]}
        for entry in accepted
    }
    test_rows = read_jsonl(bundle / "test.jsonl")
    assert {row["sample_id"] for row in test_rows} == {entry["sample_id"] for entry in accepted}
    assert meta["counts"]["test"]["excluded"] == 0
    assert meta["max_len_source"] == "train"

    vocab = json.loads((bundle / "vocab.json").read_bytes())
    names = {value: name for name, value in vocab.items()}
    runs = tmp_path / "runs"
    scores: dict[str, list[dict]] = {}
    for config, shift in (("base", 0.0), ("mask", -0.25)):
        rows = []
        for i, row in enumerate(sorted(test_rows, key=lambda r: r["sample_id"])):
            targets = row["tokens"][1:]  # Score positions exclude BOS, as in the trainer.
            nlls = [1.0 + i + pos / 16 + shift for pos in range(len(targets))]
            refs = [pos for pos, token in enumerate(targets) if names[token].startswith("REF_")]
            rows.append({
                "sample_id": row["sample_id"], "n_tokens": len(targets),
                "nll": sum(nlls), "nll_per_token": sum(nlls) / len(targets),
                "token_nll": nlls, "ref_pos": refs,
                "ref_k": [int(names[targets[pos]][4:]) for pos in refs],
                "ref_correct": [int(pos % 2 == 0) for pos in refs],
                "ref_type_nll": [0.125 for _ in refs],
            })
        scores[config] = rows
        run = runs / controlled_eval.run_dir_name("integration_", config, 10, 0)
        run.mkdir(parents=True)
        # Reverse one file to exercise pairing by sample_id rather than row order.
        write_jsonl(run / "test_scores.jsonl", rows if config == "base" else list(reversed(rows)))

    report = controlled_eval.summarize(runs, "integration_", 10, bundle,
                                       ["base", "mask"], "base", [0], by_bucket=True)
    assert report["n_test"] == len(accepted)
    for config in scores:
        buckets = report["runs"][config][0]["by_bucket"]
        assert set(buckets) == {entry["bucket"] for entry in accepted}
        for bucket, result in buckets.items():
            rows = [row for row in scores[config]
                    if index["samples"][row["sample_id"]]["bucket"] == bucket]
            n_tokens = sum(row["n_tokens"] for row in rows)
            expected_nll = sum(row["nll"] for row in rows) / n_tokens
            ref_nlls = [row["token_nll"][pos] for row in rows for pos in row["ref_pos"]]
            correct = [value for row in rows for value in row["ref_correct"]]
            assert ref_nlls
            assert result["n_samples"] == result["raw"] == result["retained"] == 2
            assert result["n_tokens"] == n_tokens
            assert result["nll_per_token"] == pytest.approx(expected_nll)
            assert result["ref_nll"] == pytest.approx(sum(ref_nlls) / len(ref_nlls))
            assert result["edge_accuracy"] == pytest.approx(sum(correct) / len(correct))
            assert "wf" not in result
            summary = report["summary"][config]["by_bucket"][bucket]
            assert summary["n_samples"]["mean"] == 2
            assert summary["nll_per_token"]["mean"] == pytest.approx(expected_nll)
            if config == "mask":
                delta = result["paired_delta_vs_baseline"]
                assert delta["n"] == 2
                assert delta["mean_delta"] == pytest.approx(-0.25)
                assert delta["ci95"] == pytest.approx([-0.25, -0.25])
                assert summary["paired_delta_vs_baseline"]["mean"] == pytest.approx(-0.25)
    json.dumps(report, allow_nan=False)

    fixture = Path(__file__).parent / "fixtures" / "generator_v1_manifest.json"
    legacy = tmp_path / "legacy"
    expected = json.loads(fixture.read_bytes())
    build_dataset(legacy, {split: tuple(info["seed_range"])
                           for split, info in expected["splits"].items()},
                  expected["generator"]["config"], expected["generator"]["version"])
    assert (legacy / "manifest.json").read_bytes() == fixture.read_bytes()
    for split, info in expected["splits"].items():
        for entry in info["samples"]:
            path = legacy / split / f"{entry['sample_id']}.json"
            mg = store.load_sample(path)
            assert mg.motifs
            assert store.encode_metagraph(mg) == json.loads(path.read_bytes())["metagraph"]


def test_v2_imports_are_lightweight() -> None:
    subprocess.run([
        sys.executable, "-c",
        "import cfg_reducer, cfg_reducer.dataset_v2, training.controlled_eval, "
        "training.prepare_tokens; import sys; "
        "assert 'torch' not in sys.modules and 'matplotlib' not in sys.modules",
    ], cwd=Path(__file__).resolve().parents[1], check=True, timeout=5)
