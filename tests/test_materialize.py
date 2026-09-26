"""mixture 実体化の契約。データ生成・学習コマンドは実行しない。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from experiments.mixture_doe import materialize as m
from training import seqdesign as sd
from training.runner.types import plan_from_json, plan_to_json, validate_plan


ROOT = Path(__file__).resolve().parents[1]


def envelope():
    space = sd.Space(("lay", "str", "spa"), ({"lay": .5, "str": .25, "spa": .25},),
                     ("lay", "str", "spa"))
    context: dict[str, object] = {"size": 24, "prefix": "a24", "data_root": "data", "max_offset": 20,
               "baseline_dir": "experiments/mixture_doe/base_target", "response_scale": "excess_target",
               "dataset_version": "abc123", "test_manifests": {t: "a" * 64 for t in ("lay", "str", "spa")}}
    campaign = sd.Campaign(1, "adaptive24", 0, 0, space, ("lay", "str", "spa"),
        {"kind": "bal", "responses": ["lay", "str", "spa"], "direction": "minimize"},
        "mixture", context, (), (), (), {"seed": 42, "draws": 0})
    # 実際の propose envelope を JSON 経由で読む。
    return json.loads(json.dumps(sd.propose(campaign, 2, 2, "random")[1]))


def read_json(path):
    return json.loads(path.read_text())


def commands(path, module):
    return [shlex.split(line) for line in path.read_text().splitlines()
            if line.startswith(f"uv run python -m {module} ")]


@pytest.mark.parametrize("weights", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.4, 0, .6), (.5, .25, .25)])
def test_spec_shapes(tmp_path, weights):
    payload = envelope()
    for row in payload["proposals"]:
        row["factors"] = dict(zip(("lay", "str", "spa"), weights))
    m.materialize(payload, tmp_path)
    spec = read_json(tmp_path / "specs/spec_p100.json")
    template = read_json(ROOT / "experiments/mixture_doe/specs/spec_p7.json")
    components = [dict(component, weight=weight)
                  for component, weight in zip(template["params"]["components"], weights) if weight]
    if len(components) == 1:
        expected = {"family": components[0]["family"], "num_nodes": 24, "params": components[0]["params"]}
    else:
        expected = dict(template, params={"components": components})
    assert spec == expected
    if weights == (1, 0, 0):
        assert spec == read_json(ROOT / "experiments/mixture_doe/specs/spec_p1.json")


def test_duplicate_point_plan_commands_and_manifest(tmp_path):
    payload = envelope()
    # 予約順と seed 順を意図的に逆にして順序保証を検証する。
    payload["proposals"].reverse()
    payload["proposals"][0]["seeds"].reverse()
    manifest = m.materialize(payload, tmp_path)
    plan = plan_from_json((tmp_path / "plan.json").read_text())
    validate_plan(plan)
    assert plan_from_json(plan_to_json(plan)) == plan
    assert plan.plan_id == "adaptive24_r1"
    assert len(plan.blocks) == 1
    block = plan.blocks[0]
    assert block.source_bundle == "data/tok_d24_p100_lay"
    assert len(block.train) == 4 and len(block.rescore) == 8
    for seed, job in enumerate(block.train):
        assert job.name == f"a24_s24_p100_mask_n24_s{seed}"
        assert (job.seed, job.sample_seed) == (seed, 1000 + seed)
        assert (job.epochs, job.patience, job.num_samples, job.constrained_samples) == (300, 20, 400, 400)
        assert job.extra == ("--ref-legal-mask",)
        for target, rescore in zip(("str", "spa"), block.rescore[2*seed:2*seed+2]):
            assert rescore.source_run == job.name
            assert rescore.target_bundle == f"data/tok_d24_p100_{target}"
            assert rescore.out_run == f"a24_s24_p1002{target}_mask_n24_s{seed}"
    gen, = commands(tmp_path / "generate.sh", "cfg_reducer.dataset_v2")
    assert gen == ["uv", "run", "python", "-m", "cfg_reducer.dataset_v2", "--spec",
        str(tmp_path / "specs/spec_p100.json"), "--out", "data/d24_p100",
        "--split", "train=800000:802150", "--split", "val=802150:802370", "--split", "test=802370:802400",
        "--exclude-dataset", "data/s24_layered", "--exclude-dataset", "data/s24_structured",
        "--exclude-dataset", "data/s24_spaghetti", "--allow-version-mismatch", "--allow-incomplete"]
    tok = commands(tmp_path / "tokenize.sh", "training.prepare_tokens")
    assert len(tok) == 3
    for args, (target, family) in zip(tok, m.TARGETS):
        assert args[args.index("--dataset")+1] == "data/d24_p100"
        assert args[args.index("--test-dataset")+1] == f"data/s24_{family}"
        assert args[args.index("--max-offset")+1] == "20"
        assert args[args.index("--out")+1] == f"data/tok_d24_p100_{target}"
    collect, = commands(tmp_path / "collect.sh", "training.mixture_doe")
    for key, value in {"--seeds": "0,1,2,3", "--points": "100", "--prefix": "a24", "--size": "24",
                       "--runs": "runs", "--data": "data", "--baseline-mode": "target",
                       "--baseline-dir": payload["context"]["baseline_dir"],
                       "--out": str(tmp_path / "obs_p100.json")}.items():
        assert collect[collect.index(key)+1] == value
    assert manifest == read_json(tmp_path / "manifest.json")
    assert set(manifest) == {"version", "campaign_id", "campaign_revision", "context", "proposal_ids", "files"}
    assert manifest["context"] == payload["context"]
    assert manifest["proposal_ids"] == ["q000002", "q000001"]
    assert set(manifest["files"]) == {"specs/spec_p100.json", "generate.sh", "tokenize.sh", "collect.sh", "plan.json"}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name in ("generate.sh", "tokenize.sh", "collect.sh"):
        text = (tmp_path / name).read_text()
        assert "set -euo pipefail\n" in text
        assert "rm " not in text
    assert "dataset_version differs from HEAD" in (tmp_path / "generate.sh").read_text()
    assert "fixed test manifest hash mismatch" in (tmp_path / "tokenize.sh").read_text()


def test_order_noncontiguous_seeds_n48_and_quoting(tmp_path):
    payload = envelope()
    payload["context"].update(size=48, max_offset=37, data_root="data 'quoted'; $(literal)",
                              baseline_dir="baseline with spaces")
    payload["proposals"][0].update(point=102, candidate_id=1, factors={"lay": 1, "str": 0, "spa": 0}, seeds=[9, 4])
    payload["proposals"][1]["seeds"] = [8, 6]
    out = tmp_path / "out 'quoted'"
    m.materialize(payload, out)
    plan = plan_from_json((out / "plan.json").read_text())
    assert [b.train[0].name for b in plan.blocks] == ["a24_s48_p100_mask_n48_s6", "a24_s48_p102_mask_n48_s4"]
    for spec_path in (out / "specs").glob("*.json"):
        spec = read_json(spec_path)
        params = ([c["params"] for c in spec["params"]["components"]]
                  if spec["family"] == "mixture" else [spec["params"]])
        assert spec["num_nodes"] == 48
        assert all(p["loop_count"] == 7 and p["goto_count"] == 4 for p in params)
    for args in commands(out / "tokenize.sh", "training.prepare_tokens"):
        assert args[args.index("--max-offset")+1] == "37"
        assert args[args.index("--dataset")+1].startswith(payload["context"]["data_root"] + "/")
    collect = commands(out / "collect.sh", "training.mixture_doe")
    assert [args[args.index("--seeds")+1] for args in collect] == ["6,8", "4,9"]
    assert collect[0][collect[0].index("--baseline-dir")+1] == "baseline with spaces"


def test_repeat_is_noop_and_conflict_is_preflighted(tmp_path):
    payload = envelope()
    m.materialize(payload, tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    m.materialize(deepcopy(payload), tmp_path)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    # 後段ファイルの衝突でも欠けた前段ファイルを書かない。
    (tmp_path / "specs/spec_p100.json").unlink()
    (tmp_path / "manifest.json").write_text("conflict")
    with pytest.raises(ValueError, match="conflicting existing file"):
        m.materialize(payload, tmp_path)
    assert not (tmp_path / "specs/spec_p100.json").exists()
    assert (tmp_path / "manifest.json").read_text() == "conflict"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(context=None),
    lambda p: p.update(version=2),
    lambda p: p.update(unexpected=True),
    lambda p: p["context"].update(size=32),
    lambda p: p["context"].update(max_offset=19),
    lambda p: p["context"].update(response_scale="raw"),
    lambda p: p["proposals"][1].update(seeds=[0]),
    lambda p: p["proposals"][1].update(factors={"lay": 1, "str": 0, "spa": 0}),
    lambda p: p["proposals"][0].update(factors={"lay": float("nan"), "str": 0, "spa": 0}),
    lambda p: p["proposals"][0].update(seeds=[True]),
])
def test_invalid_input_does_not_write(tmp_path, mutate):
    payload = envelope()
    mutate(payload)
    with pytest.raises(ValueError):
        m.materialize(payload, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_direct_script_cli(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps(envelope()))
    args = [sys.executable, str(ROOT / "experiments/mixture_doe/materialize.py"),
            "--proposal", str(proposal), "--out-dir", str(tmp_path / "out")]
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/plan.json").is_file()
    proposal.write_text("null")
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and "envelope" in result.stderr
